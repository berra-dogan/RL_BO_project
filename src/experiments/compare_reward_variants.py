"""Compare two reward variants head-to-head across every (dimension, horizon,
held_out_function) fold in a lofo_comparison.csv, using final_regret as the
primary metric and mean_total_scaled_move_cost as a tiebreaker when the
regret difference is small (noise-level).

Usage:
    python src/experiments/compare_reward_variants.py \
        optimistic_improvement_movement_cost2 optimistic_improvement_movement_cost3

Writes src/summary/compare_<reward_a>_vs_<reward_b>.csv and prints a summary.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def to_float(value):
    try:
        x = float(value)
        return None if x != x else x  # filter NaN
    except (TypeError, ValueError):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reward_a")
    parser.add_argument("reward_b")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "src" / "summary" / "lofo_comparison.csv",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--close-threshold-pct",
        type=float,
        default=5.0,
        help="Relative regret difference (%%) below which the move-cost "
        "tiebreak is used instead of trusting the regret difference.",
    )
    return parser.parse_args()


def compare(rows, reward_a, reward_b, close_threshold_pct):
    by_key = {}
    for row in rows:
        if row["reward"] not in (reward_a, reward_b):
            continue
        key = (row["dimension"], row["horizon"], row["held_out_function"])
        by_key.setdefault(key, {})[row["reward"]] = row

    results = []
    for key in sorted(by_key):
        cell = by_key[key]
        if reward_a not in cell or reward_b not in cell:
            continue
        ra, rb = cell[reward_a], cell[reward_b]
        fr_a, fr_b = to_float(ra["final_regret"]), to_float(rb["final_regret"])
        mc_a, mc_b = (
            to_float(ra["mean_total_scaled_move_cost"]),
            to_float(rb["mean_total_scaled_move_cost"]),
        )
        dt_a, dt_b = (
            to_float(ra["mean_decision_time"]),
            to_float(rb["mean_decision_time"]),
        )
        if fr_a is None or fr_b is None:
            winner, basis, diff_pct = "missing_data", "missing_data", None
        else:
            denom = max(abs(fr_a), abs(fr_b), 1e-12)
            diff_pct = 100 * (fr_a - fr_b) / denom
            if abs(diff_pct) >= close_threshold_pct:
                winner = reward_a if fr_a < fr_b else reward_b
                basis = "regret"
            elif mc_a is not None and mc_b is not None and abs(mc_a - mc_b) > 1e-9:
                winner = reward_a if mc_a < mc_b else reward_b
                basis = "move_cost_tiebreak"
            elif dt_a is not None and dt_b is not None and abs(dt_a - dt_b) > 1e-9:
                # Regret and move cost both tied -> same effective config was
                # selected for both rewards, so this is decision-time noise
                # (cluster load at run time), not a real algorithmic
                # difference. Still used as the final tiebreak per instruction.
                winner = reward_a if dt_a < dt_b else reward_b
                basis = "decision_time_tiebreak"
            else:
                winner, basis = "tie", "true_tie"

        results.append(
            {
                "dimension": key[0],
                "horizon": key[1],
                "held_out_function": key[2],
                f"final_regret_{reward_a}": fr_a,
                f"final_regret_{reward_b}": fr_b,
                "regret_diff_pct": diff_pct,
                f"mean_total_scaled_move_cost_{reward_a}": mc_a,
                f"mean_total_scaled_move_cost_{reward_b}": mc_b,
                f"mean_decision_time_{reward_a}": dt_a,
                f"mean_decision_time_{reward_b}": dt_b,
                "basis": basis,
                "winner": winner,
            }
        )
    return results


def main():
    args = parse_args()
    reward_a, reward_b = args.reward_a, args.reward_b
    output = args.output or (
        ROOT
        / "src"
        / "summary"
        / f"compare_{reward_a}_vs_{reward_b}.csv"
    )

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    results = compare(rows, reward_a, reward_b, args.close_threshold_pct)
    if not results:
        raise SystemExit(
            f"No overlapping (dimension, horizon, held_out_function) rows found "
            f"for {reward_a!r} and {reward_b!r} in {args.input}"
        )

    fieldnames = list(results[0].keys())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    tally = defaultdict(int)
    for r in results:
        tally[r["winner"]] += 1

    print(f"Wrote {len(results)} rows to {output}")
    print("Overall winner tally:")
    for key in (reward_a, reward_b, "tie", "missing_data"):
        if tally[key]:
            print(f"  {key}: {tally[key]}")


if __name__ == "__main__":
    main()
