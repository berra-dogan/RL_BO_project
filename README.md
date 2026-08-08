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

This writes a CSV file named like:

`RL_BO_<dimension>D_<function>_h<horizon>.csv`

## Notes

- Main experiment config is in `src/config.py`.
- Objective functions are in `src/objective_functions.py`.
- Reward functions are registered in `src/rewards.py`. Add a function there and include it in `REWARD_FUNCTIONS` to make it available through `--reward-mode`.
- TuRBO code is under `src/turbo/`.
- Cluster PBS submission scripts are in `cluster/`.
- Local helper scripts are in `scripts/`.
- See `docs/CLUSTER_WORKFLOW.md` for the Imperial cluster workflow.

## Editing Tuning Parameters

All tuning/test parameters for the leave-one-function-out pipeline live in two
Python files:

- `src/experiments/experiment_runner.py`: `BASE_SETTINGS` (shared experiment
  settings, including the default `dimension`), `SEARCH_SPACE` (shared PPO/GP
  hyperparameter grid), and `TUNE_BUDGET`/`TEST_BUDGET` (how many runs/BO
  iterations tuning vs. testing use).
- `src/experiments/reward_configs.py`: `REWARD_PARAM_SPACES`, the per-reward
  parameter grid (e.g. `snake_path_cost_weight` for `snake`).

The default objective dimension is `2`, meant for fast initial experimentation;
raise it in `BASE_SETTINGS["dimension"]` once the pipeline is validated.

Shell/PBS script defaults (which functions, rewards, and dimension to run by
default) are centralized in `scripts/lofo_defaults.sh` instead of being
duplicated across scripts.

## Reward Function Sweeps

Run one reward directly:

```bash
python src/main.py --reward-mode optimistic_improvement --reward-param std_weight=0.2
```

```python
SEARCH_SPACE = {
    "max_episodes": [120],
    "ppo_learning_rate": [1e-4, 2e-4],
    ...
}
```

## Leave-One-Function-Out Evaluation

Tune on four benchmark functions and test on the held-out fifth function:

```bash
python src/experiments/run_leave_one_function_out.py \
  --test-function ackley \
  --dimension 10 \
  --reward earlbo snake \
  --mode all \
  --skip-existing
```

For cluster arrays, print the flattened tuning-job count and run individual
indices with `--mode tune --index <index>`. After all indexed jobs complete,
run `--mode collect` followed by `--mode test`.

Results are grouped by dimension and held-out function:
`output/leave_one_function_out/dimension_<n>/held_out_<function>/`.

To skip tuning and test the singleton values currently defined in
`REWARD_PARAM_SPACES` and `SEARCH_SPACE`, add `--use-current-params`:

```bash
python src/experiments/run_leave_one_function_out.py \
  --test-function ackley \
  --dimension 10 \
  --reward snake earlbo \
  --mode test \
  --use-current-params
```
