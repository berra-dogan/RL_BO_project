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
- See `docs/REWARD_FINE_TUNING_ARRAY.md` for reward tuning and result collection.

## Reward Function Sweeps

Run one reward directly:

```bash
python src/main.py --reward-mode optimistic_improvement --reward-param std_weight=0.2
```

Sweep reward variants by editing `SEARCH_SPACE` in `src/experiments/run_one_reward_experiments.py`:

```python
SEARCH_SPACE = {
    "reward_mode": ["earlbo", "snake", "log_improvement", "optimistic_improvement"],
    "snake_path_cost_weight": [0.0, 0.01, 0.05],
    "reward_param_std_weight": [0.1, 0.2],
    ...
}
```

Keys named `reward_param_<name>` are forwarded as `--reward-param <name>=<value>`.
