# EARL_BO

Reinforcement-Learning-guided Bayesian Optimization experiments.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python src/main.py
```

This writes per-run CSVs and an aggregate summary named like:

`RL_BO_<dimension>D_<function>_h<horizon>.csv`

Choose the acquisition with `--acquisition {rl_bo,pure_bo}` and the reward
shaping with `--reward-mode <name>`, plus `--reward-param KEY=VALUE` (repeatable)
or `--reward-params-json '{...}'`. For example:

```bash
python src/main.py --reward-mode optimistic_improvement --reward-param std_weight=0.2
```

## Layout

- Experiment config / hyperparameter dataclasses: `src/config.py`.
- Objective functions: `src/objective_functions.py` — five smooth benchmarks
  (`ackley`, `sphere`, `sum_square`, `levy`, `rosenbrock`) plus
  `rastrigin` / `schwefel` / `michalewicz`, which are never tuned on and are
  only used as generalisation targets on their canonical domains.
- Reward functions: registered in `src/rewards.py`. Add a function and an entry
  in `REWARD_FUNCTIONS` to expose it through `--reward-mode`.
- TuRBO acquisition code: `src/turbo/`.
- Experiment pipeline: `src/experiments/`.
- Cluster PBS jobs: `cluster/`; login-node submission wrappers: `scripts/`.

## Cluster workflow

From the Mac, sync the repo up:

```bash
./scripts/sync_to_cluster.sh
```

From the cluster login node, submit the full tune → collect → test chain:

```bash
bash scripts/submit_leave_one_function_out_cluster.sh
```

Defaults (dimensions, horizons, functions, rewards) live in
`scripts/lofo_defaults.sh`. Override per run with the `DIMENSIONS` / `HORIZONS`
/ `REWARDS` environment variables, or pass held-out function names as positional
arguments.

## Editing tuning parameters

All tuning/test parameters for the leave-one-function-out pipeline live in two
Python files:

- `src/experiments/experiment_runner.py`: `BASE_SETTINGS` (shared experiment
  settings, including the fallback `dimension` and `horizon`), `SEARCH_SPACE`
  (shared PPO/GP hyperparameter grid), and `TEST_BUDGET` / `tune_budget()` (how
  many runs and BO iterations testing vs. tuning use).
- `src/experiments/reward_configs.py`: `REWARD_PARAM_SPACES`, the per-reward
  parameter grid (e.g. `snake_path_cost_weight` for `snake`).

`BASE_SETTINGS["dimension"]` / `["horizon"]` are only the fallback for a bare
`python src/experiments/run_leave_one_function_out.py`; every script passes an
explicit grid from `scripts/lofo_defaults.sh`.

## Leave-One-Function-Out evaluation

Tune on all benchmark functions except one, then test on the held-out function:

```bash
python src/experiments/run_leave_one_function_out.py \
  --test-function ackley \
  --dimension 10 \
  --horizon 3 \
  --reward earlbo snake \
  --mode all \
  --skip-existing
```

`--mode all` runs tune + collect + test in one process. On the cluster this is
split across array jobs: `--print-total` reports the flattened tuning-job count,
`--mode tune --index <i>` runs one job, and `--mode collect` then `--mode test`
aggregate and evaluate. Tuning is fold-independent — a single shared pool of
`(dimension, horizon, reward, config, function)` runs is written under
`_shared_tuning/` and reused by every held-out fold.

Per-fold results are grouped by dimension, horizon, held-out function and
reward:

```
output/leave_one_function_out/dimension_<n>/horizon_<h>/held_out_<function>/<reward>/
    tuning/best_config.json
    test_<function>/test_config.json
```

To skip tuning and test the singleton values currently defined in
`REWARD_PARAM_SPACES` / `SEARCH_SPACE`, add `--use-current-params`:

```bash
python src/experiments/run_leave_one_function_out.py \
  --test-function ackley --dimension 10 --horizon 3 \
  --reward snake earlbo --mode test --use-current-params
```

## Baselines and results aggregation

- `earlbo` and `pure_bo` are run outside the LOFO tree as flat grids
  (`scripts/submit_earlbo_grid_cluster.sh`,
  `scripts/submit_pure_bo_grid_cluster.sh`) into `output/earlbo_grid/` and
  `output/pure_bo_grid/`.
- `python src/experiments/summarize_earlbo_grid.py` regenerates
  `src/experiments/earlbo_avg_scaled_move_cost.py`, the per-`(function,
  dimension, horizon)` movement-budget calibration used by the
  budgeted-exploration rewards.
- `src/experiments/collect_all_functions.py` picks one "trained on all five
  functions" config per `(dimension, horizon, reward)` by re-ranking the
  existing tuning runs (no new experiments).
- `src/experiments/gather_lofo_results.py` (wrapper:
  `scripts/gather_lofo_results_local.sh`) collects every LOFO / all-functions /
  baseline cell into `src/summary/lofo_comparison.csv`.
