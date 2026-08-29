#!/usr/bin/env bash
# Submit cluster/collect_leave_one_function_out_single.pbs: same env-var
# interface as submit_lofo_best_configs_cluster.sh (DIMENSIONS/HORIZONS/
# REWARDS/positional functions), but runs the whole grid as ONE qsub job
# instead of one array subjob per fold. Collect is pure aggregation of
# already-finished tuning result.json files (no experiments run), so it's
# cheap enough to do in a single job and avoids waiting on many separate
# queue slots.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/lofo_defaults.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-../output/leave_one_function_out}"
read -ra DIMENSIONS <<< "${DIMENSIONS:-$LOFO_DEFAULT_DIMENSION}"
read -ra HORIZONS <<< "${HORIZONS:-$LOFO_DEFAULT_HORIZON}"
read -ra REWARDS <<< "${REWARDS:-$LOFO_DEFAULT_REWARDS}"
if [ "$#" -gt 0 ]; then
  FUNCTIONS=("$@")
else
  read -ra FUNCTIONS <<< "$LOFO_DEFAULT_FUNCTIONS"
fi

TOTAL_FOLDS=$((${#FUNCTIONS[@]} * ${#DIMENSIONS[@]} * ${#HORIZONS[@]}))
FUNCTIONS_CSV="$(IFS=:; echo "${FUNCTIONS[*]}")"
DIMENSIONS_CSV="$(IFS=:; echo "${DIMENSIONS[*]}")"
HORIZONS_CSV="$(IFS=:; echo "${HORIZONS[*]}")"
REWARDS_CSV="$(IFS=:; echo "${REWARDS[*]}")"

echo "Submitting 1 job covering $TOTAL_FOLDS folds"
echo "Dimensions: ${DIMENSIONS[*]}"
echo "Horizons: ${HORIZONS[*]}"
echo "Held-out functions: ${FUNCTIONS[*]}"
echo "Rewards: ${REWARDS[*]}"

qsub \
  -v "LOFO_FUNCTIONS=$FUNCTIONS_CSV,LOFO_DIMENSIONS=$DIMENSIONS_CSV,LOFO_HORIZONS=$HORIZONS_CSV,LOFO_REWARDS=$REWARDS_CSV,LOFO_OUTPUT_ROOT=$OUTPUT_ROOT" \
  cluster/collect_leave_one_function_out_single.pbs
