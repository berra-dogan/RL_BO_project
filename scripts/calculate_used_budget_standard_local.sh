#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" src/experiments/calculate_used_budget.py \
  --reward snake earlbo log_improvement normalized_improvement optimistic_improvement \
  --results-root output/reward_finetune_reward_params \
  --output-dir output/movement_cost_usage \
  --device cpu
