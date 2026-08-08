"""Summarize reward/objective experiment results.

This script scans an experiment output tree like:

    output/current_parameter_tests/dimension_10/
      held_out_ackley/
        earlbo/
          test_ackley/
            RL_BO_10D_ackley_h3.csv
            used_budget.json

and writes compact CSV/JSON summaries.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


RESULT_COLUMNS = [
    "dimension",
    "objective_function",
    "reward",
    "num_iterations",
    "final_avg_regret",
    "final_std_regret",
    "best_avg_regret",
    "mean_avg_regret",
    "final_avg_time",
    "final_std_time",
    "mean_total_scaled_move_cost",
    "std_total_scaled_move_cost",
    "mean_total_raw_move_cost",
    "std_total_raw_move_cost",
    "mean_budget_fraction_used",
    "result_csv",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create average result summaries for reward/objective experiments.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("../output/current_parameter_tests/dimension_10"),
        help="Directory containing held_out_<function>/<reward>/test_<function> results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summaries. Defaults to <input-root>/summary.",
    )
    return parser.parse_args()


def read_float(row, column):
    value = row.get(column, "")
    if value in ("", None):
        return None
    return float(value)


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def read_result_csv(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    avg_regrets = [read_float(row, "Avg Regret") for row in rows]
    last = rows[-1]
    return {
        "num_iterations": len(rows),
        "final_avg_regret": read_float(last, "Avg Regret"),
        "final_std_regret": read_float(last, "Std Regret"),
        "best_avg_regret": min(value for value in avg_regrets if value is not None),
        "mean_avg_regret": mean(avg_regrets),
        "final_avg_time": read_float(last, "Avg Time"),
        "final_std_time": read_float(last, "Std Time"),
    }


def read_budget_json(run_dir):
    path = run_dir / "used_budget.json"
    if not path.exists():
        return {}

    payload = json.loads(path.read_text())
    return {
        "mean_total_scaled_move_cost": payload.get("mean_total_scaled_move_cost"),
        "std_total_scaled_move_cost": payload.get("std_total_scaled_move_cost"),
        "mean_total_raw_move_cost": payload.get("mean_total_raw_move_cost"),
        "std_total_raw_move_cost": payload.get("std_total_raw_move_cost"),
        "mean_budget_fraction_used": payload.get("mean_budget_fraction_used"),
    }


def infer_metadata(input_root, result_csv):
    relative = result_csv.relative_to(input_root)
    parts = relative.parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected result path under {input_root}: {result_csv}")

    held_out = parts[0]
    reward = parts[1]
    test_dir = parts[2]

    objective = held_out.removeprefix("held_out_")
    if test_dir.startswith("test_"):
        objective = test_dir.removeprefix("test_")

    dimension = ""
    for part in input_root.parts:
        if part.startswith("dimension_"):
            dimension = part.removeprefix("dimension_")
            break

    return dimension, objective, reward


def collect_results(input_root):
    rows = []
    for result_csv in sorted(input_root.glob("held_out_*/*/test_*/RL_BO_*.csv")):
        metrics = read_result_csv(result_csv)
        if metrics is None:
            continue

        dimension, objective, reward = infer_metadata(input_root, result_csv)
        run_dir = result_csv.parent
        row = {
            "dimension": dimension,
            "objective_function": objective,
            "reward": reward,
            **metrics,
            **read_budget_json(run_dir),
            "result_csv": str(result_csv),
        }
        rows.append(row)
    return rows


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def aggregate(rows, group_column):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_column]].append(row)

    output = []
    for group_value, group_rows in sorted(grouped.items()):
        output.append({
            group_column: group_value,
            "num_results": len(group_rows),
            "avg_final_regret": mean(row["final_avg_regret"] for row in group_rows),
            "avg_best_regret": mean(row["best_avg_regret"] for row in group_rows),
            "avg_mean_regret": mean(row["mean_avg_regret"] for row in group_rows),
            "avg_final_time": mean(row["final_avg_time"] for row in group_rows),
            "avg_scaled_move_cost": mean(
                row.get("mean_total_scaled_move_cost") for row in group_rows
            ),
            "avg_budget_fraction_used": mean(
                row.get("mean_budget_fraction_used") for row in group_rows
            ),
        })
    return output


def add_objective_ranks(rows):
    by_objective = defaultdict(list)
    for row in rows:
        by_objective[row["objective_function"]].append(row)

    for objective_rows in by_objective.values():
        objective_rows.sort(key=lambda row: row["final_avg_regret"])
        for rank, row in enumerate(objective_rows, start=1):
            row["rank_within_objective"] = rank


def main():
    args = parse_args()
    input_root = args.input_root
    output_dir = args.output_dir or input_root / "summary"

    rows = collect_results(input_root)
    if not rows:
        raise SystemExit(f"No result CSV files found under {input_root}")

    add_objective_ranks(rows)

    result_columns = RESULT_COLUMNS.copy()
    result_columns.insert(4, "rank_within_objective")
    write_csv(output_dir / "reward_objective_results.csv", rows, result_columns)

    reward_rows = aggregate(rows, "reward")
    reward_rows.sort(key=lambda row: row["avg_final_regret"])
    for rank, row in enumerate(reward_rows, start=1):
        row["rank_by_avg_final_regret"] = rank
    write_csv(
        output_dir / "reward_averages.csv",
        reward_rows,
        [
            "rank_by_avg_final_regret",
            "reward",
            "num_results",
            "avg_final_regret",
            "avg_best_regret",
            "avg_mean_regret",
            "avg_final_time",
            "avg_scaled_move_cost",
            "avg_budget_fraction_used",
        ],
    )

    objective_rows = aggregate(rows, "objective_function")
    write_csv(
        output_dir / "objective_averages.csv",
        objective_rows,
        [
            "objective_function",
            "num_results",
            "avg_final_regret",
            "avg_best_regret",
            "avg_mean_regret",
            "avg_final_time",
            "avg_scaled_move_cost",
            "avg_budget_fraction_used",
        ],
    )

    overall = {
        "input_root": str(input_root),
        "num_results": len(rows),
        "num_rewards": len({row["reward"] for row in rows}),
        "num_objective_functions": len({row["objective_function"] for row in rows}),
        "avg_final_regret": mean(row["final_avg_regret"] for row in rows),
        "avg_best_regret": mean(row["best_avg_regret"] for row in rows),
        "avg_mean_regret": mean(row["mean_avg_regret"] for row in rows),
        "avg_final_time": mean(row["final_avg_time"] for row in rows),
    }
    (output_dir / "overall_summary.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n",
    )

    print(f"Wrote {len(rows)} result summaries to {output_dir}")


if __name__ == "__main__":
    main()
