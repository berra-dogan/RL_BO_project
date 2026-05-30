import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from run_experiments import (
    ROOT,
    SEARCH_SPACE,
    TUNE_BUDGET,
    build_sweep,
    copy_run_snapshot,
    run_one,
    save_json,
    score_result,
)


REWARD_PARAM_SPACE = {
    "earlbo": [{}],
    "snake": [
        {"snake_path_cost_weight": 0.0},
        {"snake_path_cost_weight": 0.01},
        {"snake_path_cost_weight": 0.05},
    ],
    "log_improvement": [
        {"reward_param_scale": 0.5},
        {"reward_param_scale": 1.0},
        {"reward_param_scale": 2.0},
    ],
    "normalized_improvement": [{}],
    "optimistic_improvement": [
        {"reward_param_std_weight": 0.1},
        {"reward_param_std_weight": 0.2},
        {"reward_param_std_weight": 0.5},
    ],
}

REWARD_NAMES = tuple(REWARD_PARAM_SPACE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run or collect one flattened reward x hyperparameter array job."
    )
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--output-root", default="reward_finetune_array")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--print-total", action="store_true")
    return parser.parse_args()


def build_reward_sweep():
    base_sweep = build_sweep()
    jobs = []
    for reward_name in REWARD_NAMES:
        reward_configs = []
        for base_config in base_sweep:
            for reward_config in REWARD_PARAM_SPACE[reward_name]:
                reward_configs.append({**base_config, **reward_config})
        for config_idx, config in enumerate(reward_configs):
            jobs.append((reward_name, config_idx, config))
    return jobs


def total_jobs():
    return len(build_reward_sweep())


def decode_index(index):
    reward_sweep = build_reward_sweep()
    if index < 0 or index >= len(reward_sweep):
        raise IndexError(f"Index {index} outside valid range 0-{total_jobs() - 1}")
    return reward_sweep[index]


def array_index(args):
    if args.index is not None:
        return args.index
    try:
        return int(os.environ["PBS_ARRAY_INDEX"])
    except KeyError as exc:
        raise RuntimeError("Missing --index and PBS_ARRAY_INDEX is not set") from exc


def runner_args(reward_name):
    return SimpleNamespace(
        reward_mode=reward_name,
        snake_path_cost_weight=None,
        reward_param=[],
        reward_params_json=None,
    )


def run_array_job(args):
    index = array_index(args)
    reward_name, config_idx, config = decode_index(index)
    output_root = (ROOT / args.output_root / reward_name).resolve()
    run_dir = output_root / "tuning" / f"config_{config_idx:03d}"
    result_path = run_dir / "result.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        run_dir / "config.json",
        {
            "array_index": index,
            "reward": reward_name,
            "config_id": config_idx,
            "params": config,
            "budget": TUNE_BUDGET,
        },
    )

    if args.skip_existing and result_path.exists():
        print(f"Result exists, skipping: {result_path}")
        return

    print(f"array_index={index}")
    print(f"reward={reward_name}")
    print(f"config_id={config_idx}")
    print(f"params={config}")

    result = run_one(
        config,
        run_dir,
        TUNE_BUDGET,
        args.device,
        args.skip_existing,
        runner_args(reward_name),
    )
    save_json(result_path, result)

    if result["status"] not in ("ok", "skipped"):
        raise RuntimeError(f"Array job failed: {result}")


def collect_reward(output_root, reward_name):
    tuning_root = output_root / reward_name / "tuning"
    rows = []
    reward_configs = [
        (config_idx, config)
        for candidate_reward, config_idx, config in build_reward_sweep()
        if candidate_reward == reward_name
    ]

    for config_idx, config in reward_configs:
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
        rows.append({"config_id": config_idx, **config, **result})

    tuning_root.mkdir(parents=True, exist_ok=True)
    results_csv = tuning_root / "tuning_results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["status"] in ("ok", "skipped")]
    if not valid_rows:
        print(f"No successful runs for {reward_name}. See {results_csv}")
        return None

    best_row = min(
        valid_rows,
        key=lambda row: (row["score"], row["final_regret"], row["best_regret"]),
    )
    best_payload = {
        "config_id": best_row["config_id"],
        "params": {
            key: best_row[key]
            for key in best_row
            if key in SEARCH_SPACE or key == "snake_path_cost_weight" or key.startswith("reward_param_")
        },
        "reward": reward_name,
        "score": best_row["score"],
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
    return best_payload


def collect(args):
    output_root = (ROOT / args.output_root).resolve()
    summary = {}
    for reward_name in REWARD_NAMES:
        best_payload = collect_reward(output_root, reward_name)
        if best_payload is not None:
            summary[reward_name] = best_payload
    save_json(output_root / "reward_summary.json", summary)
    print(f"Saved summary to {output_root / 'reward_summary.json'}")


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
