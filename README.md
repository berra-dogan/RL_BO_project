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
- TuRBO code is under `EARL_BO/turbo/`.
