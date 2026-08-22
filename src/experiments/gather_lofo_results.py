"""Gather leave-one-function-out results into a single comparison CSV.

Walks the full grid of dimension x horizon x held_out_function x reward and
pulls the tuning selection metrics (best_config.json) and held-out test
metrics (test_config.json) for each cell. Any cell missing a file, or a
field within a file, is filled with NaN so every reward has the same shape
and can be compared directly (e.g. pivoted by reward in a spreadsheet or
pandas).

Expects a tree like:
    output/leave_one_function_out/dimension_<d>/horizon_<h>/held_out_<fn>/<reward>/
        tuning/best_config.json
        test_<fn>/test_config.json

The "earlbo" and "pure_bo" rewards are never tuned/tested through that LOFO
tree (they have no held-out fold) — their results instead live in flat
grids:
    output/earlbo_grid/dimension_<d>/horizon_<h>/<fn>/
    output/pure_bo_grid/dimension_<d>/horizon_<h>/<fn>/
        RL_BO_<d>D_<fn>_h<h>.csv
        used_budget.json
so those rows are read from there instead.
"""

import argparse
import csv
import json
import math
from pathlib import Path

NAN = float("nan")

COLUMNS = [
    "dimension",
    "horizon",
    "held_out_function",
    "reward",
    "best_config_found",
    "config_id",
    "mean_rank",
    "test_config_found",
    "test_status",
    "score",
    "best_regret",
    "final_regret",
    "mean_budget_fraction_used",
    "mean_remaining_movement_budget",
    "mean_total_raw_move_cost",
    "std_total_raw_move_cost",
    "mean_total_scaled_move_cost",
    "std_total_scaled_move_cost",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gather LOFO best-config/test results across dimension, "
        "horizon, held-out function, and reward into one comparison CSV.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("output/leave_one_function_out"),
        help="Root directory containing dimension_<d>/horizon_<h>/held_out_<fn>/<reward>/.",
    )
    parser.add_argument(
        "--earlbo-root",
        type=Path,
        default=None,
        help="Root directory containing dimension_<d>/horizon_<h>/<fn>/RL_BO_*.csv "
        "for the earlbo reward. Defaults to <input-root's parent>/earlbo_grid.",
    )
    parser.add_argument(
        "--pure-bo-root",
        type=Path,
        default=None,
        help="Root directory containing dimension_<d>/horizon_<h>/<fn>/RL_BO_*.csv "
        "for the pure_bo baseline. Defaults to <input-root's parent>/pure_bo_grid.",
    )
    parser.add_argument("--dimensions", nargs="+", default=["3", "5", "10"])
    parser.add_argument("--horizons", nargs="+", default=["3", "5"])
    parser.add_argument(
        "--functions",
        nargs="+",
        default=["ackley", "sphere", "sum_square", "levy", "rosenbrock"],
    )
    parser.add_argument(
        "--rewards",
        nargs="+",
        default=[
            "earlbo",
            "pure_bo",
            "snake",
            "log_improvement",
            "normalized_improvement",
            "optimistic_improvement",
            "budgeted_exploration",
            "lookahead_budgeted_exploration",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input-root>/summary/lofo_comparison.csv.",
    )
    return parser.parse_args()


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def get(mapping, *keys, default=NAN):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return default
        mapping = mapping[key]
    return mapping if mapping is not None else default


def read_regret_column(csv_path):
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    regrets = [float(row["Avg Regret"]) for row in rows if row.get("Avg Regret")]
    if not regrets:
        return None
    return {"final_regret": regrets[-1], "best_regret": min(regrets)}


def gather_lofo_row(input_root, dimension, horizon, function_name, reward):
    reward_root = (
        input_root
        / f"dimension_{dimension}"
        / f"horizon_{horizon}"
        / f"held_out_{function_name}"
        / reward
    )
    best_config = read_json(reward_root / "tuning" / "best_config.json")
    test_config = read_json(reward_root / f"test_{function_name}" / "test_config.json")
    test_result = get(test_config, "test_result", default={}) if test_config else {}

    return {
        "dimension": dimension,
        "horizon": horizon,
        "held_out_function": function_name,
        "reward": reward,
        "best_config_found": best_config is not None,
        "config_id": get(best_config, "config_id"),
        "mean_rank": get(best_config, "mean_rank"),
        "test_config_found": test_config is not None,
        "test_status": get(test_result, "status", default="") or "",
        "score": get(test_result, "score"),
        "best_regret": get(test_result, "best_regret"),
        "final_regret": get(test_result, "final_regret"),
        "mean_budget_fraction_used": get(test_result, "mean_budget_fraction_used"),
        "mean_remaining_movement_budget": get(
            test_result, "mean_remaining_movement_budget"
        ),
        "mean_total_raw_move_cost": get(test_result, "mean_total_raw_move_cost"),
        "std_total_raw_move_cost": get(test_result, "std_total_raw_move_cost"),
        "mean_total_scaled_move_cost": get(test_result, "mean_total_scaled_move_cost"),
        "std_total_scaled_move_cost": get(test_result, "std_total_scaled_move_cost"),
    }


def gather_flat_grid_row(grid_root, dimension, horizon, function_name, reward):
    function_root = (
        grid_root / f"dimension_{dimension}" / f"horizon_{horizon}" / function_name
    )
    summary_csv = function_root / f"RL_BO_{dimension}D_{function_name}_h{horizon}.csv"
    budget = read_json(function_root / "used_budget.json") or {}
    regret = read_regret_column(summary_csv) if summary_csv.exists() else None

    return {
        "dimension": dimension,
        "horizon": horizon,
        "held_out_function": function_name,
        "reward": reward,
        "best_config_found": False,
        "config_id": NAN,
        "mean_rank": NAN,
        "test_config_found": regret is not None,
        "test_status": "ok" if regret is not None else "",
        "score": get(regret or {}, "final_regret"),
        "best_regret": get(regret or {}, "best_regret"),
        "final_regret": get(regret or {}, "final_regret"),
        "mean_budget_fraction_used": get(budget, "mean_budget_fraction_used"),
        "mean_remaining_movement_budget": get(budget, "mean_remaining_movement_budget"),
        "mean_total_raw_move_cost": get(budget, "mean_total_raw_move_cost"),
        "std_total_raw_move_cost": get(budget, "std_total_raw_move_cost"),
        "mean_total_scaled_move_cost": get(budget, "mean_total_scaled_move_cost"),
        "std_total_scaled_move_cost": get(budget, "std_total_scaled_move_cost"),
    }


def gather_rows(
    input_root, flat_grid_roots, dimensions, horizons, functions, rewards
):
    rows = []
    for dimension in dimensions:
        for horizon in horizons:
            for function_name in functions:
                for reward in rewards:
                    if reward in flat_grid_roots:
                        rows.append(
                            gather_flat_grid_row(
                                flat_grid_roots[reward],
                                dimension,
                                horizon,
                                function_name,
                                reward,
                            )
                        )
                    else:
                        rows.append(
                            gather_lofo_row(
                                input_root, dimension, horizon, function_name, reward
                            )
                        )
    return rows


def format_value(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_value(row.get(column)) for column in columns})


def main():
    args = parse_args()
    output = args.output or args.input_root / "summary" / "lofo_comparison.csv"
    earlbo_root = args.earlbo_root or args.input_root.parent / "earlbo_grid"
    pure_bo_root = args.pure_bo_root or args.input_root.parent / "pure_bo_grid"
    flat_grid_roots = {"earlbo": earlbo_root, "pure_bo": pure_bo_root}

    rows = gather_rows(
        args.input_root,
        flat_grid_roots,
        args.dimensions,
        args.horizons,
        args.functions,
        args.rewards,
    )
    write_csv(output, rows, COLUMNS)

    total = len(rows)
    found_best = sum(1 for row in rows if row["best_config_found"])
    found_test = sum(1 for row in rows if row["test_config_found"])
    print(f"Wrote {total} rows to {output}")
    print(f"best_config.json found: {found_best}/{total}")
    print(f"test_config.json found: {found_test}/{total}")


if __name__ == "__main__":
    main()
