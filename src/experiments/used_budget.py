"""Read and write movement-budget summaries for experiment runs."""

import csv
import json
import os
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np


MOVEMENT_RESULT_COLUMNS = [
    "Avg Scaled Move Cost",
    "Std Scaled Move Cost",
    "Avg Raw Move Cost",
    "Std Raw Move Cost",
    "Avg Cumulative Scaled Move Cost",
    "Std Cumulative Scaled Move Cost",
    "Avg Cumulative Raw Move Cost",
    "Std Cumulative Raw Move Cost",
]


def ensure_movement_cost_columns(summary_path):
    """Upgrade an existing aggregate CSV from its adjacent per-run CSVs."""
    summary_path = Path(summary_path)
    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        original_columns = list(reader.fieldnames or [])
        summary_rows = list(reader)

    if all(column in original_columns for column in MOVEMENT_RESULT_COLUMNS):
        return False

    run_rows = []
    for run_path in sorted(summary_path.parent.glob("run_*.csv")):
        with run_path.open(newline="") as handle:
            run_rows.append(list(csv.DictReader(handle)))

    if not summary_rows or not run_rows:
        raise ValueError(f"Cannot add movement costs without run data beside {summary_path}")
    if any(len(rows) != len(summary_rows) for rows in run_rows):
        raise ValueError(f"Run and summary row counts differ beside {summary_path}")

    cumulative_scaled = [0.0] * len(run_rows)
    cumulative_raw = [0.0] * len(run_rows)
    for iteration, summary_row in enumerate(summary_rows):
        scaled = []
        raw = []
        scaled_cumulative = []
        raw_cumulative = []
        for run_index, rows in enumerate(run_rows):
            run_row = rows[iteration]
            scaled_value = float(run_row.get("Scaled Move Cost", 0.0))
            raw_value = float(run_row.get("Raw Move Cost", 0.0))
            cumulative_scaled[run_index] += scaled_value
            cumulative_raw[run_index] += raw_value
            scaled.append(scaled_value)
            raw.append(raw_value)
            scaled_cumulative.append(cumulative_scaled[run_index])
            raw_cumulative.append(cumulative_raw[run_index])

        summary_row.update({
            "Avg Scaled Move Cost": fmean(scaled),
            "Std Scaled Move Cost": pstdev(scaled),
            "Avg Raw Move Cost": fmean(raw),
            "Std Raw Move Cost": pstdev(raw),
            "Avg Cumulative Scaled Move Cost": fmean(scaled_cumulative),
            "Std Cumulative Scaled Move Cost": pstdev(scaled_cumulative),
            "Avg Cumulative Raw Move Cost": fmean(raw_cumulative),
            "Std Cumulative Raw Move Cost": pstdev(raw_cumulative),
        })

    base_columns = [
        column for column in original_columns
        if column not in MOVEMENT_RESULT_COLUMNS
    ]
    insertion_index = (
        base_columns.index("Std Regret") + 1
        if "Std Regret" in base_columns
        else len(base_columns)
    )
    columns = (
        base_columns[:insertion_index]
        + MOVEMENT_RESULT_COLUMNS
        + base_columns[insertion_index:]
    )
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)
    os.replace(temporary_path, summary_path)
    print(f"Movement costs added to existing result CSV: {summary_path}")
    return True


def write_used_budget_summary(output_dir: Path, movement_budget):
    run_summaries = []
    for path in sorted(output_dir.glob("run_*.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue

        total_scaled = sum(float(row.get("Scaled Move Cost", 0.0)) for row in rows)
        total_raw = sum(float(row.get("Raw Move Cost", 0.0)) for row in rows)
        run_summaries.append({
            "run_id": int(path.stem.removeprefix("run_")),
            "total_scaled_move_cost": total_scaled,
            "total_raw_move_cost": total_raw,
            "remaining_movement_budget": (
                max(0.0, movement_budget - total_scaled)
                if movement_budget is not None
                else None
            ),
        })

    if not run_summaries:
        return None

    scaled_totals = np.array(
        [run["total_scaled_move_cost"] for run in run_summaries],
        dtype=float,
    )
    raw_totals = np.array(
        [run["total_raw_move_cost"] for run in run_summaries],
        dtype=float,
    )
    payload = {
        "movement_budget": movement_budget,
        "num_runs": len(run_summaries),
        "mean_total_scaled_move_cost": float(np.mean(scaled_totals)),
        "std_total_scaled_move_cost": float(np.std(scaled_totals)),
        "mean_total_raw_move_cost": float(np.mean(raw_totals)),
        "std_total_raw_move_cost": float(np.std(raw_totals)),
        "mean_remaining_movement_budget": (
            float(np.mean([run["remaining_movement_budget"] for run in run_summaries]))
            if movement_budget is not None
            else None
        ),
        "mean_budget_fraction_used": (
            float(np.mean(np.minimum(scaled_totals / movement_budget, 1.0)))
            if movement_budget is not None and movement_budget > 0
            else None
        ),
        "runs": run_summaries,
    }
    summary_path = output_dir / "used_budget.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Used budget summary saved to {summary_path}")
    return summary_path


def read_used_budget(run_dir):
    summary_path = Path(run_dir) / "used_budget.json"
    if not summary_path.exists():
        return {}

    payload = json.loads(summary_path.read_text())
    return {
        "used_budget_json": str(summary_path),
        "mean_total_scaled_move_cost": payload["mean_total_scaled_move_cost"],
        "std_total_scaled_move_cost": payload["std_total_scaled_move_cost"],
        "mean_total_raw_move_cost": payload["mean_total_raw_move_cost"],
        "std_total_raw_move_cost": payload["std_total_raw_move_cost"],
        "mean_remaining_movement_budget": payload.get("mean_remaining_movement_budget"),
        "mean_budget_fraction_used": payload.get("mean_budget_fraction_used"),
    }
