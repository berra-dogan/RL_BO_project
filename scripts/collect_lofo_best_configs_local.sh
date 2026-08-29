#!/usr/bin/env bash
# Run only the LOFO "collect" stage locally: aggregates already-finished
# tuning result.json files into best_config.json for each
# dimension/horizon/held-out-function/reward combo. Does not run any
# experiments (no --mode test), so it's cheap and safe to run on Mac as
# long as the tuning output is already present under output/leave_one_function_out.
#
# Defaults to REWARDS=lookahead_budgeted_exploration; override any of
# DIMENSIONS/HORIZONS/FUNCTIONS/REWARDS via env vars, e.g.:
#   REWARDS="snake earlbo" bash scripts/collect_lofo_best_configs_local.sh
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/lofo_defaults.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-../output/leave_one_function_out}"
read -ra DIMENSIONS <<< "${DIMENSIONS:-$LOFO_DEFAULT_DIMENSION}"
read -ra HORIZONS <<< "${HORIZONS:-$LOFO_DEFAULT_HORIZON}"
read -ra FUNCTIONS <<< "${FUNCTIONS:-$LOFO_DEFAULT_FUNCTIONS}"
read -ra REWARDS <<< "${REWARDS:-lookahead_budgeted_exploration}"

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

echo "dimensions=${DIMENSIONS[*]}"
echo "horizons=${HORIZONS[*]}"
echo "functions=${FUNCTIONS[*]}"
echo "rewards=${REWARDS[*]}"
echo "output_root=$OUTPUT_ROOT"
echo "python=$PYTHON_BIN"

failures=0
expected_count=0
best_count=0

for dimension in "${DIMENSIONS[@]}"; do
  for horizon in "${HORIZONS[@]}"; do
    for function_name in "${FUNCTIONS[@]}"; do
      expected_count=$((expected_count + ${#REWARDS[@]}))
      echo "[collect] dimension=$dimension horizon=$horizon held_out=$function_name"
      if ! "$PYTHON_BIN" src/experiments/run_leave_one_function_out.py \
        --test-function "$function_name" \
        --dimension "$dimension" \
        --horizon "$horizon" \
        --reward "${REWARDS[@]}" \
        --mode collect \
        --output-root "$OUTPUT_ROOT"; then
        echo "[error] Collection failed for dimension=$dimension horizon=$horizon held_out=$function_name" >&2
        failures=$((failures + 1))
        continue
      fi

      for reward in "${REWARDS[@]}"; do
        best_path="output/leave_one_function_out/dimension_${dimension}/horizon_${horizon}/held_out_${function_name}/${reward}/tuning/best_config.json"
        [ -f "$best_path" ] && best_count=$((best_count + 1))
      done
    done
  done
done

echo "[summary] best_configs=$best_count/$expected_count"

if [ "$failures" -ne 0 ] || [ "$best_count" -ne "$expected_count" ]; then
  echo "[warning] Collection finished with incomplete results. Check warnings.jsonl files." >&2
  find output/leave_one_function_out -name warnings.jsonl -print >&2
  exit 1
fi

echo "[done] All best configurations were collected."
