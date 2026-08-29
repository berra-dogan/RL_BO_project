# Cluster PBS Scripts

For the full command checklist for syncing, tuning, collecting, testing, and
resyncing results, see `docs/CLUSTER_COMMANDS.md`.

Every script here implements the leave-one-function-out (LOFO) pipeline: tune
each reward on all-but-one objective function, collect the best config per
fold, then test it on the held-out function. Default dimension/functions/
rewards are defined once in `scripts/lofo_defaults.sh` — edit that file to
change what runs by default, or override with the `DIMENSIONS`/`FUNCTIONS`/
`REWARDS` environment variables shown below.

## Leave-One-Function-Out Evaluation

Submit tuning, collection, and held-out testing for every objective function:

```bash
bash scripts/submit_leave_one_function_out_cluster.sh
```

Pass function names to run only selected held-out folds:

```bash
bash scripts/submit_leave_one_function_out_cluster.sh ackley levy
```

Dimension 2 is used by default (see `scripts/lofo_defaults.sh`). Submit
multiple dimensions with:

```bash
DIMENSIONS="2 10" bash scripts/submit_leave_one_function_out_cluster.sh
```

Select a subset of rewards with:

```bash
REWARDS="earlbo snake" bash scripts/submit_leave_one_function_out_cluster.sh
```

The submission helper creates one tuning array and a dependent collection/test
array. Per-fold results (best config, held-out test) are grouped under
`output/leave_one_function_out/dimension_<n>/horizon_<h>/held_out_<function>/<reward>/`.

A tuning run for a given `(dimension, horizon, reward, config, training
function)` is identical regardless of which function is held out, so the tuning
stage writes one shared, fold-independent pool at
`output/leave_one_function_out/dimension_<n>/horizon_<h>/_shared_tuning/<reward>/config_<id>/<function>/`
and every fold's collection stage reads from it. This is why the tuning array is
sized `dimensions x horizons x tuning-jobs` with no fold axis (previously each
fold re-ran the same configs, ~4x redundant for 5 functions).

To run just one stage (e.g. resubmitting after a failure), use
`scripts/submit_lofo_tuning_cluster.sh`,
`scripts/submit_lofo_best_configs_cluster.sh`, or
`scripts/submit_lofo_tests_cluster.sh`, or submit the underlying `.pbs` files
directly:

- `run_leave_one_function_out_tuning.pbs`
- `collect_leave_one_function_out.pbs`
- `test_leave_one_function_out.pbs`

To skip tuning and test the current singleton parameters for every
function/reward pair in parallel:

```bash
qsub cluster/test_current_params_all_functions.pbs
```

This creates one array element per function/reward pair and writes under
`output/current_parameter_tests/dimension_<n>/`.

## Movement Cost

Tuning and test jobs save `used_budget.json` automatically.
