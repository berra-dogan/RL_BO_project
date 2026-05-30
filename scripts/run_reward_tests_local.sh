#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x "venv/bin/python" ]; then
  echo "Missing local interpreter: venv/bin/python" >&2
  echo "Create/install the local venv before running local tests." >&2
  exit 1
fi

PYTHON_BIN="venv/bin/python"
OUTPUT_ROOT="${OUTPUT_ROOT:-reward_finetune_reward_params}"

run_reward_test() {
  local reward="$1"
  shift

  local best_config="EARL_BO/${OUTPUT_ROOT}/${reward}/tuning/best_config.json"
  if [ ! -f "$best_config" ]; then
    echo "[skip] Missing tuned config: $best_config"
    return 0
  fi

  echo "[test] reward=${reward}"
  "$PYTHON_BIN" EARL_BO/run_experiments.py \
    --mode test \
    --device cpu \
    --output-root "${OUTPUT_ROOT}/${reward}" \
    --reward-mode "$reward" \
    "$@" \
    --skip-existing
}

run_reward_test earlbo
run_reward_test snake --snake-path-cost-weight 0.01
run_reward_test log_improvement
run_reward_test normalized_improvement
run_reward_test optimistic_improvement --reward-param std_weight=0.2

echo "[done] Test outputs:"
find "EARL_BO/${OUTPUT_ROOT}" -path '*/test_best/test_config.json' -print
