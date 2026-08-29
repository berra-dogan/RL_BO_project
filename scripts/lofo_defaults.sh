#!/usr/bin/env bash
# Single place to edit the default dimension/horizon/functions/rewards for
# every leave-one-function-out script (local and cluster). Source this file,
# then read from LOFO_DEFAULT_DIMENSION / LOFO_DEFAULT_HORIZON /
# LOFO_DEFAULT_FUNCTIONS / LOFO_DEFAULT_REWARDS.

LOFO_DEFAULT_DIMENSION="3 5 10"
LOFO_DEFAULT_HORIZON="3 5"
LOFO_DEFAULT_FUNCTIONS="ackley sphere sum_square levy rosenbrock"
LOFO_DEFAULT_REWARDS="earlbo snake log_improvement log_improvement_movement_cost log_improvement_movement_cost2 optimistic_improvement optimistic_improvement_movement_cost optimistic_improvement_movement_cost2 budgeted_exploration lookahead_budgeted_exploration"
