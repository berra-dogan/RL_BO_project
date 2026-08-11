#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/lofo_defaults.sh

DIMENSION="${DIMENSION:-$LOFO_DEFAULT_DIMENSION}"
HORIZON="${HORIZON:-$LOFO_DEFAULT_HORIZON}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../output/leave_one_function_out}"
read -ra FUNCTIONS <<< "${FUNCTIONS:-$LOFO_DEFAULT_FUNCTIONS}"
read -ra REWARDS <<< "${REWARDS:-$LOFO_DEFAULT_REWARDS}"

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

RESULT_ROOT="output/leave_one_function_out/dimension_${DIMENSION}/horizon_${HORIZON}"
if [ ! -d "$RESULT_ROOT" ]; then
  echo "Missing tuning results: $RESULT_ROOT" >&2
  echo "Sync the cluster outputs locally before running this script." >&2
  exit 1
fi

echo "dimension=$DIMENSION"
echo "horizon=$HORIZON"
echo "functions=${FUNCTIONS[*]}"
echo "rewards=${REWARDS[*]}"
echo "output_root=$OUTPUT_ROOT"
echo "python=$PYTHON_BIN"

failures=0

for function_name in "${FUNCTIONS[@]}"; do
  echo "[collect] held_out=$function_name"
  if ! "$PYTHON_BIN" src/experiments/run_leave_one_function_out.py \
    --test-function "$function_name" \
    --dimension "$DIMENSION" \
    --horizon "$HORIZON" \
    --reward "${REWARDS[@]}" \
    --mode collect \
    --output-root "$OUTPUT_ROOT"; then
    echo "[error] Collection failed for held_out=$function_name" >&2
    failures=$((failures + 1))
    continue
  fi

  echo "[test] held_out=$function_name"
  if ! "$PYTHON_BIN" src/experiments/run_leave_one_function_out.py \
    --test-function "$function_name" \
    --dimension "$DIMENSION" \
    --horizon "$HORIZON" \
    --reward "${REWARDS[@]}" \
    --mode test \
    --device cpu \
    --output-root "$OUTPUT_ROOT" \
    --skip-existing; then
    echo "[error] Testing failed for held_out=$function_name" >&2
    failures=$((failures + 1))
  fi
done

expected_count=$((${#FUNCTIONS[@]} * ${#REWARDS[@]}))
best_count=0
test_count=0
for function_name in "${FUNCTIONS[@]}"; do
  for reward in "${REWARDS[@]}"; do
    reward_root="$RESULT_ROOT/held_out_${function_name}/${reward}"
    [ -f "$reward_root/tuning/best_config.json" ] && best_count=$((best_count + 1))
    [ -f "$reward_root/test_${function_name}/test_config.json" ] && test_count=$((test_count + 1))
  done
done

echo "[summary] best_configs=$best_count/$expected_count"
echo "[summary] test_configs=$test_count/$expected_count"

if [ "$failures" -ne 0 ] || [ "$best_count" -ne "$expected_count" ] || [ "$test_count" -ne "$expected_count" ]; then
  echo "[warning] Workflow finished with incomplete results. Check warnings.jsonl files." >&2
  find "$RESULT_ROOT" -name warnings.jsonl -print >&2
  exit 1
fi

echo "[done] All best configurations were collected and tested."
