#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

FUNCTIONS_DEFAULT="ackley sphere sum_square levy rosenbrock"
DIMENSIONS_DEFAULT="3 5 10"
HORIZONS_DEFAULT="3 5"

read -ra FUNCTIONS <<< "${FUNCTIONS:-$FUNCTIONS_DEFAULT}"
read -ra DIMENSIONS <<< "${DIMENSIONS:-$DIMENSIONS_DEFAULT}"
read -ra HORIZONS <<< "${HORIZONS:-$HORIZONS_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/pure_bo_grid}"

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

TOTAL_CELLS=$(( ${#FUNCTIONS[@]} * ${#DIMENSIONS[@]} * ${#HORIZONS[@]} ))
echo "Running $TOTAL_CELLS pure BO grid cells locally"
echo "Functions: ${FUNCTIONS[*]}"
echo "Dimensions: ${DIMENSIONS[*]}"
echo "Horizons: ${HORIZONS[*]}"
echo "Output root: $OUTPUT_ROOT"
echo "Python: $PYTHON_BIN"

failures=0
cell=0
for DIMENSION in "${DIMENSIONS[@]}"; do
  for HORIZON in "${HORIZONS[@]}"; do
    for TEST_FUNCTION in "${FUNCTIONS[@]}"; do
      cell=$((cell + 1))
      OUTPUT_DIR="$OUTPUT_ROOT/dimension_${DIMENSION}/horizon_${HORIZON}/${TEST_FUNCTION}"
      echo "[$cell/$TOTAL_CELLS] dimension=$DIMENSION horizon=$HORIZON function=$TEST_FUNCTION"
      if ! "$PYTHON_BIN" src/main.py \
        --test-func "$TEST_FUNCTION" \
        --dimension "$DIMENSION" \
        --horizon "$HORIZON" \
        --acquisition pure_bo \
        --lower-bound -1 \
        --upper-bound 1 \
        --num-initial-data "$((5 * DIMENSION))" \
        --num-runs 3 \
        --num-experiments 50 \
        --device cpu \
        --output-dir "$OUTPUT_DIR"; then
        echo "[error] pure_bo failed for dimension=$DIMENSION horizon=$HORIZON function=$TEST_FUNCTION" >&2
        failures=$((failures + 1))
      fi
    done
  done
done

echo "[summary] $((TOTAL_CELLS - failures))/$TOTAL_CELLS cells completed"
if [ "$failures" -ne 0 ]; then
  exit 1
fi
