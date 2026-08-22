#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/lofo_defaults.sh

INPUT_ROOT="${INPUT_ROOT:-output/leave_one_function_out}"
EARLBO_ROOT="${EARLBO_ROOT:-output/earlbo_grid}"
PURE_BO_ROOT="${PURE_BO_ROOT:-output/pure_bo_grid}"
OUTPUT="${OUTPUT:-$INPUT_ROOT/summary/lofo_comparison.csv}"
read -ra DIMENSIONS <<< "${DIMENSIONS:-$LOFO_DEFAULT_DIMENSION}"
read -ra HORIZONS <<< "${HORIZONS:-$LOFO_DEFAULT_HORIZON}"
read -ra FUNCTIONS <<< "${FUNCTIONS:-$LOFO_DEFAULT_FUNCTIONS}"
# pure_bo is never tuned through the LOFO pipeline (it's a fixed analytic
# baseline, not a --reward-mode), so it isn't in LOFO_DEFAULT_REWARDS; add it
# here unless the caller passed an explicit REWARDS list.
read -ra REWARDS <<< "${REWARDS:-$LOFO_DEFAULT_REWARDS pure_bo}"

if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
elif [ -x "venv/bin/python" ]; then
  PYTHON_BIN="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  echo "No project Python interpreter found." >&2
  echo "Set PYTHON=/path/to/python or create venv/ or .venv/." >&2
  exit 1
fi

"$PYTHON_BIN" src/experiments/gather_lofo_results.py \
  --input-root "$INPUT_ROOT" \
  --earlbo-root "$EARLBO_ROOT" \
  --pure-bo-root "$PURE_BO_ROOT" \
  --dimensions "${DIMENSIONS[@]}" \
  --horizons "${HORIZONS[@]}" \
  --functions "${FUNCTIONS[@]}" \
  --rewards "${REWARDS[@]}" \
  --output "$OUTPUT"
