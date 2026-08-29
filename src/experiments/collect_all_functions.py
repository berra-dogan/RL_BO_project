"""Select the reward config that ranks best averaged over ALL benchmark functions.

`run_leave_one_function_out.py` holds one function out per fold. This script
reuses the *same* per-function tuning runs but ranks configs across every
function in FUNCTION_NAMES at once, producing a single "trained on all 5
functions" config per (dimension, horizon, reward).

No experiments are run: it only aggregates existing tuning ``result.json`` /
summary CSV files, so it is cheap enough for one short job.

A tuning run for a given (dimension, horizon, reward, config, function) is
identical regardless of which function a LOFO fold held out, so results are read
from whichever layout is present:

  <root>/dimension_<d>/horizon_<h>/_shared_tuning/<reward>/config_<id>/<fn>/
  <root>/dimension_<d>/horizon_<h>/held_out_<other>/<reward>/tuning/config_<id>/<fn>/

Writes (never touches the inputs):

  <out>/dimension_<d>/horizon_<h>/all_functions/<reward>/tuning/best_config.json
  <out>/dimension_<d>/horizon_<h>/all_functions/<reward>/tuning/all_functions_results.csv
  <out>/dimension_<d>/horizon_<h>/all_functions/reward_summary.json
"""

import argparse
import csv
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.experiment_runner import ROOT, save_json
from experiments.reward_configs import (
    REWARD_NAMES,
    base_settings,
    load_result,
    reward_configs,
)
from experiments.run_leave_one_function_out import (
    FUNCTION_NAMES,
    average_ranks,
    selection_score,
)


DEFAULT_REWARDS = (
    "snake",
    "log_improvement",
    "optimistic_improvement",
    "budgeted_exploration",
    "lookahead_budgeted_exploration",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dimensions", type=int, nargs="+", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", required=True)
    parser.add_argument(
        "--rewards",
        nargs="+",
        choices=REWARD_NAMES,
        default=list(DEFAULT_REWARDS),
        help="Rewards to select a config for (default: %(default)s).",
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        choices=FUNCTION_NAMES,
        default=list(FUNCTION_NAMES),
        help="Functions to rank across (default: all five).",
    )
    parser.add_argument("--input-root", default="../output/leave_one_function_out")
    parser.add_argument("--output-root", default="../output/leave_one_function_out")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a (dimension, horizon, reward) whose best_config.json already exists.",
    )
    return parser.parse_args()


def dimension_horizon_root(root, dimension, horizon):
    return (
        ROOT / root / f"dimension_{dimension}" / f"horizon_{horizon}"
    ).resolve()


def find_run_dir(input_root, reward, config_id, function_name):
    """Return an existing tuning-run directory for this (reward, config, function)."""
    shared = (
        input_root
        / "_shared_tuning"
        / reward
        / f"config_{config_id:03d}"
        / function_name
    )
    if shared.is_dir():
        return shared
    folds = sorted(
        input_root.glob(
            f"held_out_*/{reward}/tuning/config_{config_id:03d}/{function_name}"
        )
    )
    for fold in folds:
        if fold.is_dir():
            return fold
    return None


def collect_reward(input_root, output_root, reward, functions, dimension, horizon):
    configs = reward_configs(reward)
    rows = [{"config_id": index, **config} for index, config in enumerate(configs)]
    missing = 0

    for function_name in functions:
        valid_indices = []
        scores = []
        for config_id, row in enumerate(rows):
            run_dir = find_run_dir(input_root, reward, config_id, function_name)
            result = load_result(run_dir) if run_dir is not None else {"status": "missing"}
            row[f"{function_name}_status"] = result["status"]
            if result["status"] in ("ok", "skipped"):
                score = selection_score(reward, result)
                row[f"{function_name}_score"] = score
                valid_indices.append(config_id)
                scores.append(score)
            else:
                missing += 1

        for config_id, rank in zip(valid_indices, average_ranks(scores)):
            rows[config_id][f"{function_name}_rank"] = rank

    for row in rows:
        ranks = [row.get(f"{name}_rank") for name in functions]
        row["mean_rank"] = (
            sum(ranks) / len(ranks) if all(rank is not None for rank in ranks) else None
        )

    tuning_root = output_root / "all_functions" / reward / "tuning"
    tuning_root.mkdir(parents=True, exist_ok=True)
    results_csv = tuning_root / "all_functions_results.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["mean_rank"] is not None]
    if not valid_rows:
        print(
            f"[warn] {reward} (d={dimension}, h={horizon}): no config has results "
            f"for all {len(functions)} functions; skipped"
        )
        return None

    best = min(valid_rows, key=lambda row: (row["mean_rank"], row["config_id"]))
    settings = base_settings(reward, configs[best["config_id"]])
    settings["dimension"] = dimension
    settings["horizon"] = horizon
    settings["test_func"] = None
    payload = {
        "selection": "all_functions",
        "reward": reward,
        "functions": list(functions),
        "selection_metric": "mean_within_function_rank",
        "config_id": best["config_id"],
        "mean_rank": best["mean_rank"],
        "function_scores": {name: best.get(f"{name}_score") for name in functions},
        "function_ranks": {name: best.get(f"{name}_rank") for name in functions},
        "params": configs[best["config_id"]],
        "base_settings": settings,
        "complete_configs": len(valid_rows),
        "total_configs": len(rows),
    }
    save_json(tuning_root / "best_config.json", payload)
    note = "" if len(valid_rows) == len(rows) else f" ({len(rows) - len(valid_rows)} incomplete)"
    print(
        f"{reward} (d={dimension}, h={horizon}): config {best['config_id']} "
        f"mean_rank={best['mean_rank']:.3f}{note}"
    )
    if missing:
        print(f"[warn] {reward}: {missing} (config, function) tuning results missing/failed")
    return payload


def main():
    args = parse_args()
    total = 0
    written = 0
    for dimension in args.dimensions:
        for horizon in args.horizons:
            input_root = dimension_horizon_root(args.input_root, dimension, horizon)
            output_root = dimension_horizon_root(args.output_root, dimension, horizon)
            if not input_root.is_dir():
                print(f"[skip] {input_root} does not exist")
                continue
            print(f"=== dimension {dimension}, horizon {horizon} ===")
            summary = {}
            for reward in args.rewards:
                total += 1
                best_path = (
                    output_root / "all_functions" / reward / "tuning" / "best_config.json"
                )
                if args.skip_existing and best_path.exists():
                    print(f"{reward} (d={dimension}, h={horizon}): exists, skipping")
                    continue
                result = collect_reward(
                    input_root, output_root, reward, args.functions, dimension, horizon
                )
                if result is not None:
                    summary[reward] = result
                    written += 1
            if summary:
                save_json(
                    output_root / "all_functions" / "reward_summary.json", summary
                )

    print(f"\n[done] wrote {written}/{total} best_config.json files")


if __name__ == "__main__":
    main()
