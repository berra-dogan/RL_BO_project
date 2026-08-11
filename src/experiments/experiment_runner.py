"""Shared configuration, execution, and scoring for reward experiments."""

import csv
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.used_budget import ensure_movement_cost_columns, read_used_budget


ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = ROOT / "main.py"

# Single place to edit tuning/test parameters for the leave-one-function-out
# pipeline. BASE_SETTINGS/SEARCH_SPACE define the shared hyperparameter grid;
# per-reward parameter grids live in experiments/reward_configs.py.
BASE_SETTINGS = {
    "dimension": 2,
    "test_func": "ackley",
    "horizon": 3,
    "lower_bound": -1.0,
    "upper_bound": 1.0,
    # num_initial_data is dimension-dependent (5 * dimension); computed in
    # effective_base_settings() rather than fixed here.
    "update_episode": 10,
    "no_improvement_threshold": 8,
    "ppo_action_std_min": 0.01,
    "ppo_k_epochs": 30,
    "ppo_eps_clip": 0.2,
    "ppo_gamma_increase": 1.0,
    "ppo_vf_coeff": 0.5,
    "ppo_freeze_num": 2,
    "gpr_rbf_length_scale": 1.0,
    "gpr_wk_noise_level": 1.0,
    "gpr_restarts": 2,
    "snake_path_cost_weight": 0.0,
}

SEARCH_SPACE = {
    "max_episodes": [120],
    "off_policy_episodes": [20],
    "encoder_learning_rate": [1e-3],
    "ppo_learning_rate": [1e-4],
    "ppo_action_std": [0.05],
    "ppo_action_decay": [0.99],
    "ppo_gamma": [0.95],
    "ppo_entropy_coeff": [0.01],
    "gpr_alpha": [1e-8],
}

TEST_BUDGET = {"num_runs": 3, "num_experiments": 50}


def tune_budget(dimension):
    return {"num_runs": 1, "num_experiments": 10 * dimension}


def build_sweep():
    keys = list(SEARCH_SPACE)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(SEARCH_SPACE[key] for key in keys))
    ]


def effective_base_settings(args):
    settings = dict(BASE_SETTINGS)
    settings["reward_mode"] = args.reward_mode
    if getattr(args, "dimension", None) is not None:
        settings["dimension"] = args.dimension
    settings["num_initial_data"] = 5 * settings["dimension"]
    if getattr(args, "horizon", None) is not None:
        settings["horizon"] = args.horizon
    if getattr(args, "test_func", None) is not None:
        settings["test_func"] = args.test_func
    if args.snake_path_cost_weight is not None:
        settings["snake_path_cost_weight"] = args.snake_path_cost_weight
    if args.movement_budget is not None:
        settings["movement_budget"] = args.movement_budget
    return settings


def summary_path(output_dir, args):
    settings = effective_base_settings(args)
    filename = (
        f"RL_BO_{settings['dimension']}D_"
        f"{settings['test_func']}_h{settings['horizon']}.csv"
    )
    return output_dir / filename


def base_command(output_dir, budget, device, args):
    command = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--num-runs",
        str(budget["num_runs"]),
        "--num-experiments",
        str(budget["num_experiments"]),
    ]
    for key, value in effective_base_settings(args).items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    if device is not None:
        command.extend(["--device", device])
    return command


def config_flags(config):
    flags = []
    for key, value in config.items():
        if key == "reward_params":
            flags.extend(["--reward-params-json", json.dumps(value, sort_keys=True)])
        elif key.startswith("reward_param_"):
            flags.extend(["--reward-param", f"{key.removeprefix('reward_param_')}={value}"])
        else:
            flags.extend([f"--{key.replace('_', '-')}", str(value)])
    return flags


def score_result(csv_path):
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in summary CSV: {csv_path}")
    regrets = [float(row["Avg Regret"]) for row in rows]
    tail = regrets[-min(5, len(regrets)):]
    return {
        "score": sum(tail) / len(tail),
        "final_regret": regrets[-1],
        "best_regret": min(regrets),
    }


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def copy_run_snapshot(run_dir, target_dir, metadata):
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(run_dir, target_dir)
    save_json(target_dir / "metadata.json", metadata)
    return target_dir


def run_one(config, run_dir, budget, device, skip_existing, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    result_csv = summary_path(run_dir, args)

    if skip_existing and result_csv.exists():
        try:
            ensure_movement_cost_columns(result_csv)
        except Exception as exc:
            print(f"[warn] Could not add movement costs to {result_csv}: {exc}")
        try:
            return {
                "status": "skipped",
                **score_result(result_csv),
                **read_used_budget(run_dir),
                "summary_csv": str(result_csv),
                "run_dir": str(run_dir),
            }
        except Exception as exc:
            print(f"[warn] Existing result invalid at {result_csv}: {exc}. Re-running.")

    completed = subprocess.run(
        base_command(run_dir, budget, device, args) + config_flags(config),
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "summary_csv": str(result_csv),
            "run_dir": str(run_dir),
        }
    if not result_csv.exists():
        return {
            "status": "missing_summary",
            "summary_csv": str(result_csv),
            "run_dir": str(run_dir),
        }

    try:
        metrics = score_result(result_csv)
    except Exception as exc:
        return {
            "status": "invalid_summary",
            "error": str(exc),
            "summary_csv": str(result_csv),
            "run_dir": str(run_dir),
        }
    return {
        "status": "ok",
        **metrics,
        **read_used_budget(run_dir),
        "summary_csv": str(result_csv),
        "run_dir": str(run_dir),
    }
