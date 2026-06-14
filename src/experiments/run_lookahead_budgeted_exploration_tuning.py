import argparse
import csv
import itertools
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from run_one_reward_experiments import (
    BASE_SETTINGS,
    ROOT,
    SEARCH_SPACE,
    copy_run_snapshot,
    run_one,
    save_json,
    score_result,
)


REWARD_NAME = "lookahead_budgeted_exploration"
LOOKAHEAD_BASE_CONFIG = {key: values[0] for key, values in SEARCH_SPACE.items()}
LOOKAHEAD_TUNE_BUDGET = {"num_runs": 1, "num_experiments": 20}

LOOKAHEAD_PARAM_SPACE = {
    "movement_budget": [5],
    "reward_param_explore_weight": [5.0],
    "reward_param_path_cost_weight": [0.05, 0.1],
    "reward_param_future_path_cost_weight": [0.005, 0.01, 0.05],
    "reward_param_future_optimism_weight": [0.5, 1.0, 2.0],
    "reward_param_future_softmax_temperature": [1.0],
    "reward_param_over_budget_penalty": [0.0],
}

BUDGET_SCORE_MOVE_WEIGHT = 0.1


def lookahead_base_settings(config):
    settings = dict(BASE_SETTINGS)
    settings["reward_mode"] = REWARD_NAME
    if "movement_budget" in config:
        settings["movement_budget"] = config["movement_budget"]
    return settings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune only the lookahead_budgeted_exploration reward."
    )
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--output-root", default="lookahead_budgeted_exploration_budget_grid")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--print-total", action="store_true")
    return parser.parse_args()


def build_lookahead_sweep():
    jobs = []
    keys = list(LOOKAHEAD_PARAM_SPACE.keys())
    values = [LOOKAHEAD_PARAM_SPACE[key] for key in keys]
    for combo in itertools.product(*values):
        reward_config = dict(zip(keys, combo))
        jobs.append({**LOOKAHEAD_BASE_CONFIG, **reward_config})
    return jobs


def total_jobs():
    return len(build_lookahead_sweep())


def array_index(args):
    if args.index is not None:
        return args.index
    try:
        return int(os.environ["PBS_ARRAY_INDEX"])
    except KeyError as exc:
        raise RuntimeError("Missing --index and PBS_ARRAY_INDEX is not set") from exc


def runner_args():
    return SimpleNamespace(
        reward_mode=REWARD_NAME,
        snake_path_cost_weight=None,
        movement_budget=None,
        reward_param=[],
        reward_params_json=None,
    )


def run_array_job(args):
    sweep = build_lookahead_sweep()
    index = array_index(args)
    if index < 0 or index >= len(sweep):
        raise IndexError(f"Index {index} outside valid range 0-{total_jobs() - 1}")

    config = sweep[index]
    output_root = (ROOT / args.output_root / REWARD_NAME).resolve()
    run_dir = output_root / "tuning" / f"config_{index:03d}"
    result_path = run_dir / "result.json"
    config_path = run_dir / "config.json"
    expected_config = {
        "array_index": index,
        "reward": REWARD_NAME,
        "config_id": index,
        "params": config,
        "budget": LOOKAHEAD_TUNE_BUDGET,
    }

    run_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and result_path.exists():
        if config_path.exists() and json.loads(config_path.read_text()) == expected_config:
            print(f"Result exists, skipping: {result_path}")
            return
        raise RuntimeError(
            "Existing result does not match the current lookahead grid. "
            f"Use a fresh --output-root or remove stale directory: {run_dir}"
        )

    save_json(config_path, expected_config)

    print(f"array_index={index}")
    print(f"reward={REWARD_NAME}")
    print(f"params={config}")

    result = run_one(
        config,
        run_dir,
        LOOKAHEAD_TUNE_BUDGET,
        args.device,
        args.skip_existing,
        runner_args(),
    )
    save_json(result_path, result)

    if result["status"] not in ("ok", "skipped"):
        raise RuntimeError(f"Array job failed: {result}")


def collect(args):
    output_root = (ROOT / args.output_root / REWARD_NAME).resolve()
    tuning_root = output_root / "tuning"
    rows = []

    for config_idx, config in enumerate(build_lookahead_sweep()):
        run_dir = tuning_root / f"config_{config_idx:03d}"
        result_path = run_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result_csvs = sorted(run_dir.glob("RL_BO_*D_*_h*.csv"))
            if not result_csvs:
                rows.append({"config_id": config_idx, **config, "status": "missing"})
                continue
            result = {
                "status": "ok",
                **score_result(result_csvs[0]),
                "summary_csv": str(result_csvs[0]),
                "run_dir": str(run_dir),
            }
        row = {"config_id": config_idx, **config, **result}
        row.update(movement_metrics(run_dir))
        row["budget_score"] = budget_score(row)
        rows.append(row)

    tuning_root.mkdir(parents=True, exist_ok=True)
    results_csv = tuning_root / "tuning_results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["status"] in ("ok", "skipped")]
    if not valid_rows:
        raise RuntimeError(f"No successful {REWARD_NAME} runs. See {results_csv}")

    best_row = min(
        valid_rows,
        key=lambda row: (row["budget_score"], row["final_regret"], row["best_regret"]),
    )
    best_payload = {
        "config_id": best_row["config_id"],
        "params": {
            key: best_row[key]
            for key in best_row
            if key in SEARCH_SPACE or key == "movement_budget" or key.startswith("reward_param_")
        },
        "base_settings": lookahead_base_settings(best_row),
        "reward": REWARD_NAME,
        "score": best_row["score"],
        "budget_score": best_row["budget_score"],
        "mean_total_scaled_move_cost": best_row.get("mean_total_scaled_move_cost"),
        "mean_total_raw_move_cost": best_row.get("mean_total_raw_move_cost"),
        "final_regret": best_row["final_regret"],
        "best_regret": best_row["best_regret"],
        "summary_csv": best_row["summary_csv"],
        "run_dir": best_row["run_dir"],
    }

    best_dir = tuning_root / "best_tuning"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    copy_run_snapshot(Path(best_row["run_dir"]), best_dir, best_payload)
    best_payload["saved_run_dir"] = str(best_dir)
    best_payload["saved_summary_csv"] = str(best_dir / Path(best_row["summary_csv"]).name)
    save_json(tuning_root / "best_config.json", best_payload)
    save_json(output_root / "lookahead_budgeted_exploration_summary.json", best_payload)
    print(f"Saved best config to {tuning_root / 'best_config.json'}")


def movement_metrics(run_dir):
    totals_scaled = []
    totals_raw = []
    for run_csv in sorted(Path(run_dir).glob("run_*.csv")):
        with run_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "Scaled Move Cost" not in rows[0] or "Raw Move Cost" not in rows[0]:
            continue
        totals_scaled.append(sum(float(row["Scaled Move Cost"]) for row in rows))
        totals_raw.append(sum(float(row["Raw Move Cost"]) for row in rows))

    if not totals_scaled:
        return {
            "mean_total_scaled_move_cost": None,
            "mean_total_raw_move_cost": None,
        }

    return {
        "mean_total_scaled_move_cost": sum(totals_scaled) / len(totals_scaled),
        "mean_total_raw_move_cost": sum(totals_raw) / len(totals_raw),
    }


def budget_score(row):
    move_cost = row.get("mean_total_scaled_move_cost")
    if move_cost is None:
        return row["score"]
    return row["score"] + BUDGET_SCORE_MOVE_WEIGHT * move_cost


def main():
    args = parse_args()
    if args.print_total:
        print(total_jobs())
        return

    if args.collect:
        collect(args)
        return

    run_array_job(args)


if __name__ == "__main__":
    main()
