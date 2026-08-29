#!/usr/bin/env bash
# Submit cluster/one_tune_tests.pbs: test-only ( --use-current-params ) runs for
# rewards whose parameter space has a single config, so tuning/collection are
# skipped. Run from the cluster login node.
#
#   REWARDS="log_improvement_movement_cost2 optimistic_improvement_movement_cost2" \
#     bash scripts/one_tune_tests_cluster.sh
#
# REWARDS is space-separated. DIMENSIONS / HORIZONS override the grid (default
# from scripts/lofo_defaults.sh); held-out functions come from positional args
# or the LOFO defaults. Results land in output/leave_one_function_out/.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/lofo_defaults.sh

REWARDS="${REWARDS:-log_improvement_movement_cost2 optimistic_improvement_movement_cost2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../output/leave_one_function_out}"
read -ra REWARDS_ARR <<< "$REWARDS"
read -ra DIMENSIONS <<< "${DIMENSIONS:-$LOFO_DEFAULT_DIMENSION}"
read -ra HORIZONS <<< "${HORIZONS:-$LOFO_DEFAULT_HORIZON}"
if [ "$#" -gt 0 ]; then
  FUNCTIONS=("$@")
else
  read -ra FUNCTIONS <<< "$LOFO_DEFAULT_FUNCTIONS"
fi

TOTAL_JOBS=$((${#FUNCTIONS[@]} * ${#HORIZONS[@]} * ${#DIMENSIONS[@]}))
LAST_INDEX=$((TOTAL_JOBS - 1))
REWARDS_CSV="$(IFS=:; echo "${REWARDS_ARR[*]}")"
FUNCTIONS_CSV="$(IFS=:; echo "${FUNCTIONS[*]}")"
DIMENSIONS_CSV="$(IFS=:; echo "${DIMENSIONS[*]}")"
HORIZONS_CSV="$(IFS=:; echo "${HORIZONS[*]}")"

echo "Submitting $TOTAL_JOBS one-config test tasks (each runs every reward)"
echo "Rewards: ${REWARDS_ARR[*]}"
echo "Dimensions: ${DIMENSIONS[*]}"
echo "Horizons: ${HORIZONS[*]}"
echo "Held-out functions: ${FUNCTIONS[*]}"

qsub \
  -J "0-$LAST_INDEX" \
  -v "REWARDS=$REWARDS_CSV,TEST_FUNCTIONS=$FUNCTIONS_CSV,TEST_DIMENSIONS=$DIMENSIONS_CSV,TEST_HORIZONS=$HORIZONS_CSV,LOFO_OUTPUT_ROOT=$OUTPUT_ROOT" \
  cluster/one_tune_tests.pbs
