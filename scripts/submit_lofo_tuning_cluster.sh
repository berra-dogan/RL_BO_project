#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/lofo_defaults.sh

PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../output/leave_one_function_out}"
read -ra DIMENSIONS <<< "${DIMENSIONS:-$LOFO_DEFAULT_DIMENSION}"
read -ra HORIZONS <<< "${HORIZONS:-$LOFO_DEFAULT_HORIZON}"
read -ra REWARDS <<< "${REWARDS:-$LOFO_DEFAULT_REWARDS}"
if [ "$#" -gt 0 ]; then
  FUNCTIONS=("$@")
else
  read -ra FUNCTIONS <<< "$LOFO_DEFAULT_FUNCTIONS"
fi

REWARD_ARGS=(--reward "${REWARDS[@]}")
# Tuning is fold-independent: one shared pool of (reward, config, function) runs,
# so the array size does not multiply by the number of held-out folds.
JOBS_PER_HORIZON=$(
  "$PYTHON" src/experiments/run_leave_one_function_out.py \
    "${REWARD_ARGS[@]}" \
    --print-total
)
TOTAL_JOBS=$((JOBS_PER_HORIZON * ${#DIMENSIONS[@]} * ${#HORIZONS[@]}))
LAST_INDEX=$((TOTAL_JOBS - 1))
FUNCTIONS_CSV="$(IFS=:; echo "${FUNCTIONS[*]}")"
DIMENSIONS_CSV="$(IFS=:; echo "${DIMENSIONS[*]}")"
HORIZONS_CSV="$(IFS=:; echo "${HORIZONS[*]}")"
REWARDS_CSV="$(IFS=:; echo "${REWARDS[*]}")"

echo "Submitting $TOTAL_JOBS tuning tasks"
echo "Dimensions: ${DIMENSIONS[*]}"
echo "Horizons: ${HORIZONS[*]}"
echo "Held-out functions: ${FUNCTIONS[*]}"
echo "Rewards: ${REWARDS[*]}"

qsub \
  -J "0-$LAST_INDEX" \
  -v "LOFO_FUNCTIONS=$FUNCTIONS_CSV,LOFO_DIMENSIONS=$DIMENSIONS_CSV,LOFO_HORIZONS=$HORIZONS_CSV,LOFO_REWARDS=$REWARDS_CSV,LOFO_OUTPUT_ROOT=$OUTPUT_ROOT" \
  cluster/run_leave_one_function_out_tuning.pbs
