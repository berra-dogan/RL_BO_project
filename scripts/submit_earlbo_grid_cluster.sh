#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

FUNCTIONS_DEFAULT="ackley sphere sum_square levy rosenbrock"
DIMENSIONS_DEFAULT="3 5 10"
HORIZONS_DEFAULT="3 5"

read -ra FUNCTIONS <<< "${FUNCTIONS:-$FUNCTIONS_DEFAULT}"
read -ra DIMENSIONS <<< "${DIMENSIONS:-$DIMENSIONS_DEFAULT}"
read -ra HORIZONS <<< "${HORIZONS:-$HORIZONS_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/earlbo_grid}"

TOTAL_JOBS=$(( ${#FUNCTIONS[@]} * ${#DIMENSIONS[@]} * ${#HORIZONS[@]} ))
LAST_INDEX=$((TOTAL_JOBS - 1))
FUNCTIONS_CSV="$(IFS=:; echo "${FUNCTIONS[*]}")"
DIMENSIONS_CSV="$(IFS=:; echo "${DIMENSIONS[*]}")"
HORIZONS_CSV="$(IFS=:; echo "${HORIZONS[*]}")"

echo "Submitting $TOTAL_JOBS earlbo test jobs"
echo "Functions: ${FUNCTIONS[*]}"
echo "Dimensions: ${DIMENSIONS[*]}"
echo "Horizons: ${HORIZONS[*]}"
echo "Output root: $OUTPUT_ROOT"

qsub \
  -J "0-$LAST_INDEX" \
  -v "EARLBO_FUNCTIONS=$FUNCTIONS_CSV,EARLBO_DIMENSIONS=$DIMENSIONS_CSV,EARLBO_HORIZONS=$HORIZONS_CSV,EARLBO_OUTPUT_ROOT=$OUTPUT_ROOT" \
  cluster/test_earlbo_grid.pbs
