#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

RESULTS_ROOT="${RESULTS_ROOT:-output/budgeted_exploration_budget_grid}"

echo "Using budgeted results root: $RESULTS_ROOT"

"$PYTHON_BIN" src/experiments/calculate_used_budget.py \
  --reward budgeted_exploration \
  --results-root "$RESULTS_ROOT" \
  --output-dir output/budgeted_exploration_movement_cost \
  --device cpu
