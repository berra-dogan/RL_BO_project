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
REWARD_ARGS=(--reward "${REWARDS[@]}")

if [ "$#" -gt 0 ]; then
  FUNCTIONS=("$@")
else
  read -ra FUNCTIONS <<< "$LOFO_DEFAULT_FUNCTIONS"
fi

FUNCTIONS_CSV="$(IFS=:; echo "${FUNCTIONS[*]}")"
DIMENSIONS_CSV="$(IFS=:; echo "${DIMENSIONS[*]}")"
HORIZONS_CSV="$(IFS=:; echo "${HORIZONS[*]}")"
REWARDS_CSV="$(IFS=:; echo "${REWARDS[*]}")"
# Tuning is fold-independent: one shared pool of (reward, config, function) runs
# is reused by every held-out fold, so the tuning array is only
# (dimension x horizon x tuning-jobs) with no fold axis.
TUNING_JOBS_PER_HORIZON=$(
  "$PYTHON" src/experiments/run_leave_one_function_out.py \
    "${REWARD_ARGS[@]}" \
    --print-total
)
TOTAL_JOBS=$((TUNING_JOBS_PER_HORIZON * ${#DIMENSIONS[@]} * ${#HORIZONS[@]}))
LAST_JOB_INDEX=$((TOTAL_JOBS - 1))
FINAL_JOB_COUNT=$((${#FUNCTIONS[@]} * ${#DIMENSIONS[@]} * ${#HORIZONS[@]}))
LAST_FINAL_INDEX=$((FINAL_JOB_COUNT - 1))
TEST_JOB_COUNT=$((${#FUNCTIONS[@]} * ${#REWARDS[@]} * ${#DIMENSIONS[@]} * ${#HORIZONS[@]}))
LAST_TEST_INDEX=$((TEST_JOB_COUNT - 1))

echo "Held-out functions: ${FUNCTIONS[*]}"
echo "Dimensions: ${DIMENSIONS[*]}"
echo "Horizons: ${HORIZONS[*]}"
echo "Rewards: ${REWARDS[*]}"
echo "Tuning jobs per dimension/horizon: $TUNING_JOBS_PER_HORIZON"
echo "Total tuning jobs (shared pool, fold-independent): $TOTAL_JOBS"
echo "Output root: $OUTPUT_ROOT"

TUNING_JOB_ID=$(
  qsub \
    -J "0-$LAST_JOB_INDEX" \
    -v "LOFO_FUNCTIONS=$FUNCTIONS_CSV,LOFO_DIMENSIONS=$DIMENSIONS_CSV,LOFO_HORIZONS=$HORIZONS_CSV,LOFO_REWARDS=$REWARDS_CSV,LOFO_OUTPUT_ROOT=$OUTPUT_ROOT" \
    cluster/run_leave_one_function_out_tuning.pbs
)
echo "Submitted tuning array: $TUNING_JOB_ID"

COLLECT_JOB_ID=$(
  qsub \
    -J "0-$LAST_FINAL_INDEX" \
    -W "depend=afteranyarray:$TUNING_JOB_ID" \
    -v "LOFO_FUNCTIONS=$FUNCTIONS_CSV,LOFO_DIMENSIONS=$DIMENSIONS_CSV,LOFO_HORIZONS=$HORIZONS_CSV,LOFO_REWARDS=$REWARDS_CSV,LOFO_OUTPUT_ROOT=$OUTPUT_ROOT" \
    cluster/collect_leave_one_function_out.pbs
)
echo "Submitted dependent collection array: $COLLECT_JOB_ID"

TEST_JOB_ID=$(
  qsub \
    -J "0-$LAST_TEST_INDEX" \
    -W "depend=afteranyarray:$COLLECT_JOB_ID" \
    -v "LOFO_FUNCTIONS=$FUNCTIONS_CSV,LOFO_DIMENSIONS=$DIMENSIONS_CSV,LOFO_HORIZONS=$HORIZONS_CSV,LOFO_REWARDS=$REWARDS_CSV,LOFO_OUTPUT_ROOT=$OUTPUT_ROOT" \
    cluster/test_leave_one_function_out.pbs
)
echo "Submitted dependent test array: $TEST_JOB_ID"
