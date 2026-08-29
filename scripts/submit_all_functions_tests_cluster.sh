#!/usr/bin/env bash
# Submit cluster/test_all_functions.pbs: evaluate the "trained on all 5
# functions" config (from collect_all_functions.py) on each benchmark function.
# One array element per (dimension, horizon, function, reward). Writes
#   output/leave_one_function_out/dimension_<n>/horizon_<h>/all_functions/<reward>/test_<function>/test_config.json
#
# Same env-var interface as submit_lofo_tests_cluster.sh:
#   DIMENSIONS / HORIZONS / REWARDS (space-separated), positional args = functions.
#   REWARDS="snake earlbo" DIMENSIONS="3 10" \
#     bash scripts/submit_all_functions_tests_cluster.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/lofo_defaults.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-../output/leave_one_function_out}"
DEFAULT_FUNCTIONS="rastrigin schwefel michalewicz"
DEFAULT_REWARDS="snake log_improvement optimistic_improvement budgeted_exploration lookahead_budgeted_exploration"
read -ra DIMENSIONS <<< "${DIMENSIONS:-$LOFO_DEFAULT_DIMENSION}"
read -ra HORIZONS <<< "${HORIZONS:-$LOFO_DEFAULT_HORIZON}"
read -ra REWARDS <<< "${REWARDS:-$DEFAULT_REWARDS}"
if [ "$#" -gt 0 ]; then
  FUNCTIONS=("$@")
else
  read -ra FUNCTIONS <<< "$DEFAULT_FUNCTIONS"
fi

TOTAL_JOBS=$((${#FUNCTIONS[@]} * ${#REWARDS[@]} * ${#DIMENSIONS[@]} * ${#HORIZONS[@]}))
LAST_INDEX=$((TOTAL_JOBS - 1))
FUNCTIONS_CSV="$(IFS=:; echo "${FUNCTIONS[*]}")"
DIMENSIONS_CSV="$(IFS=:; echo "${DIMENSIONS[*]}")"
HORIZONS_CSV="$(IFS=:; echo "${HORIZONS[*]}")"
REWARDS_CSV="$(IFS=:; echo "${REWARDS[*]}")"

echo "Submitting $TOTAL_JOBS all-functions test tasks"
echo "Dimensions: ${DIMENSIONS[*]}"
echo "Horizons: ${HORIZONS[*]}"
echo "Functions: ${FUNCTIONS[*]}"
echo "Rewards: ${REWARDS[*]}"

qsub \
  -J "0-$LAST_INDEX" \
  -v "LOFO_FUNCTIONS=$FUNCTIONS_CSV,LOFO_DIMENSIONS=$DIMENSIONS_CSV,LOFO_HORIZONS=$HORIZONS_CSV,LOFO_REWARDS=$REWARDS_CSV,LOFO_OUTPUT_ROOT=$OUTPUT_ROOT" \
  cluster/test_all_functions.pbs
