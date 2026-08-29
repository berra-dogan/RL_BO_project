#!/usr/bin/env bash
# Evaluate the "trained on all 5 fundamental functions" config
# (collect_all_functions.py -> all_functions/<reward>/tuning/best_config.json) on
# the three harder, never-tuned-on objective functions:
#   rastrigin schwefel michalewicz
# This is a pure generalisation test - none of these were seen during tuning or
# selection.
#
# Submits ONE PBS array job (cluster/test_all_functions.pbs), one element per
# (dimension, horizon, function, reward). Writes
#   output/leave_one_function_out/dimension_<n>/horizon_<h>/all_functions/<reward>/test_<function>/test_config.json
#
# REWARDS defaults to the movement-budget-free rewards, because
# budgeted_exploration / lookahead_budgeted_exploration need an earlbo
# movement-cost calibration (src/experiments/earlbo_avg_scaled_move_cost.py) for
# each (function, dimension, horizon) and none exists for these three functions.
# To include them, first run scripts/submit_earlbo_grid_cluster.sh for the new
# functions, regenerate that file, then pass REWARDS explicitly.
#
# Overrides (space-separated env vars); positional args replace the function list:
#   REWARDS="snake" DIMENSIONS="10" HORIZONS="3" \
#     bash scripts/submit_all_functions_complex_tests_cluster.sh schwefel
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/lofo_defaults.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-../output/leave_one_function_out}"
DEFAULT_FUNCTIONS="rastrigin schwefel michalewicz"
DEFAULT_REWARDS="snake log_improvement optimistic_improvement"
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

echo "Submitting $TOTAL_JOBS all-functions complex-target test tasks"
echo "Dimensions: ${DIMENSIONS[*]}"
echo "Horizons: ${HORIZONS[*]}"
echo "Functions: ${FUNCTIONS[*]}"
echo "Rewards: ${REWARDS[*]}"

qsub \
  -J "0-$LAST_INDEX" \
  -v "LOFO_FUNCTIONS=$FUNCTIONS_CSV,LOFO_DIMENSIONS=$DIMENSIONS_CSV,LOFO_HORIZONS=$HORIZONS_CSV,LOFO_REWARDS=$REWARDS_CSV,LOFO_OUTPUT_ROOT=$OUTPUT_ROOT" \
  cluster/test_all_functions.pbs
