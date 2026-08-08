#!/usr/bin/env bash
# Single place to edit the default dimension/functions/rewards for every
# leave-one-function-out script (local and cluster). Source this file, then
# read from LOFO_DEFAULT_DIMENSION / LOFO_DEFAULT_FUNCTIONS / LOFO_DEFAULT_REWARDS.

LOFO_DEFAULT_DIMENSION="2"
LOFO_DEFAULT_FUNCTIONS="ackley sphere sum_square levy rosenbrock"
LOFO_DEFAULT_REWARDS="earlbo snake log_improvement normalized_improvement optimistic_improvement budgeted_exploration lookahead_budgeted_exploration"
