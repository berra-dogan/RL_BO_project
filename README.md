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
python EARL_BO/main.py
```

This writes a CSV file named like:

`RL_BO_<dimension>D_<function>_h<horizon>.csv`

## Notes

- Main experiment config is in `EARL_BO/config.py`.
- Objective functions are in `EARL_BO/objective_functions.py`.
- Reward functions are registered in `EARL_BO/rewards.py`. Add a function there and include it in `REWARD_FUNCTIONS` to make it available through `--reward-mode`.
- TuRBO code is under `EARL_BO/turbo/`.
- Cluster usage helpers are in `scripts/`.
- See `CLUSTER_WORKFLOW.md` for the Imperial cluster workflow.

## Reward Function Sweeps

Run one reward directly:

```bash
python EARL_BO/main.py --reward-mode optimistic_improvement --reward-param std_weight=0.2
```

Sweep reward variants by editing `SEARCH_SPACE` in `EARL_BO/run_experiments.py`:

```python
SEARCH_SPACE = {
    "reward_mode": ["earlbo", "snake", "log_improvement", "optimistic_improvement"],
    "snake_path_cost_weight": [0.0, 0.01, 0.05],
    "reward_param_std_weight": [0.1, 0.2],
    ...
}
```

Keys named `reward_param_<name>` are forwarded as `--reward-param <name>=<value>`.
