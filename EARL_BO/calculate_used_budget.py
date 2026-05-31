import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from config import ExperimentConfig, GPRConfig, PPOConfig, RLBOConfig
from main import make_run_start
from objective_functions import ObjectiveFunctions
from rl_bo import RL_BO
from utils import Scaler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-run selected best configs and calculate movement cost usage."
    )
    parser.add_argument(
        "--best-config",
        action="append",
        default=[],
        help="Path under EARL_BO, or an absolute path, to a best_config.json. Can be repeated.",
    )
    parser.add_argument(
        "--reward",
        nargs="+",
        default=["snake"],
        help="Reward name(s) whose best_config.json should be loaded from --results-root.",
    )
    parser.add_argument("--results-root", default="reward_finetune")
    parser.add_argument("--output-dir", default="movement_cost_usage")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-runs", type=int, default=None)
    return parser.parse_args()


def resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent / candidate


def apply_settings(payload, device, num_runs_override):
    base = payload["base_settings"]
    params = payload["params"]

    gpr_config = GPRConfig()
    gpr_config.rbf_length_scale = base.get("gpr_rbf_length_scale", gpr_config.rbf_length_scale)
    gpr_config.wk_noise_level = base.get("gpr_wk_noise_level", gpr_config.wk_noise_level)
    gpr_config.n_restarts_optimizer = base.get("gpr_restarts", gpr_config.n_restarts_optimizer)
    gpr_config.alpha = params.get("gpr_alpha", gpr_config.alpha)

    config = ExperimentConfig(device=device)
    config.dimension = base.get("dimension", config.dimension)
    config.test_func_name = base.get("test_func", config.test_func_name)
    config.num_runs = num_runs_override or base.get("num_runs", 1)
    config.num_experiments = base.get("num_experiments", 5)
    config.num_initial_data = base.get("num_initial_data", config.num_initial_data)
    config.lower_bound = base.get("lower_bound", config.lower_bound)
    config.upper_bound = base.get("upper_bound", config.upper_bound)
    config.horizon = base.get("horizon", config.horizon)
    config.gpr_config = gpr_config

    rlbo_config = RLBOConfig()
    rlbo_config.max_episodes = params.get("max_episodes", rlbo_config.max_episodes)
    rlbo_config.update_episode = base.get("update_episode", rlbo_config.update_episode)
    rlbo_config.off_policy_episodes = params.get("off_policy_episodes", rlbo_config.off_policy_episodes)
    rlbo_config.no_improvement_threshold = base.get(
        "no_improvement_threshold",
        rlbo_config.no_improvement_threshold,
    )
    rlbo_config.encoder_learning_rate = params.get(
        "encoder_learning_rate",
        rlbo_config.encoder_learning_rate,
    )
    reward_name = payload.get("reward") or base.get("reward_mode") or "snake"
    rlbo_config.reward_mode = reward_name
    rlbo_config.snake_path_cost_weight = params.get(
        "snake_path_cost_weight",
        base.get("snake_path_cost_weight", rlbo_config.snake_path_cost_weight),
    )
    rlbo_config.movement_budget = params.get("movement_budget", base.get("movement_budget"))
    rlbo_config.reward_params = {
        key.removeprefix("reward_param_"): value
        for key, value in params.items()
        if key.startswith("reward_param_")
    }

    ppo_config = PPOConfig()
    ppo_config.learning_rate = params.get("ppo_learning_rate", ppo_config.learning_rate)
    ppo_config.action_std = params.get("ppo_action_std", ppo_config.action_std)
    ppo_config.action_std_min = base.get("ppo_action_std_min", ppo_config.action_std_min)
    ppo_config.action_decay = params.get("ppo_action_decay", ppo_config.action_decay)
    ppo_config.K_epochs = base.get("ppo_k_epochs", ppo_config.K_epochs)
    ppo_config.eps_clip = base.get("ppo_eps_clip", ppo_config.eps_clip)
    ppo_config.gamma = params.get("ppo_gamma", ppo_config.gamma)
    ppo_config.gamma_increase = base.get("ppo_gamma_increase", ppo_config.gamma_increase)
    ppo_config.VF_coeff = base.get("ppo_vf_coeff", ppo_config.VF_coeff)
    ppo_config.Entropy_coeff = params.get("ppo_entropy_coeff", ppo_config.Entropy_coeff)
    ppo_config.freeze_num = base.get("ppo_freeze_num", ppo_config.freeze_num)

    return config, rlbo_config, ppo_config


def run_trial_with_budget(seed, x_start, policy_num, config, rlbo_config, ppo_config):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(seed)

    obj_funcs = ObjectiveFunctions(config.dimension)
    test_func = obj_funcs.functions[config.test_func_name]
    x_train = x_start
    y_train = test_func.real(x_start).reshape(-1, 1)

    gpr_config = config.gpr_config
    kernel = RBF(
        length_scale=gpr_config.rbf_length_scale,
        length_scale_bounds=gpr_config.rbf_length_scale_bounds,
    ) + WhiteKernel(
        noise_level=gpr_config.wk_noise_level,
        noise_level_bounds=gpr_config.wk_noise_level_bounds,
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=gpr_config.alpha,
        n_restarts_optimizer=gpr_config.n_restarts_optimizer,
    )
    x_scaler, y_scaler = Scaler(), Scaler()
    acq_func = RL_BO(rlbo_config, ppo_config=ppo_config, device=config.device)

    lower_bound = np.full((1, config.dimension), config.lower_bound)
    upper_bound = np.full((1, config.dimension), config.upper_bound)
    rows = []
    total_scaled_move_cost = 0.0
    total_raw_move_cost = 0.0
    movement_budget_remaining = rlbo_config.movement_budget
    decision_times = []

    for iteration in range(config.num_experiments):
        y_scaler.fit_transform(y_train)
        x_scaled = x_scaler.fit_transform(x_train)
        scaled_lower_bound = ((lower_bound - x_scaler.mu) / x_scaler.std).reshape(-1)
        scaled_upper_bound = ((upper_bound - x_scaler.mu) / x_scaler.std).reshape(-1)
        prev_x_raw = x_train[-1].reshape(-1)
        prev_x_scaled = x_scaled[-1].reshape(-1)

        start_t = time.time()
        x_next_scaled = acq_func.evaluate(
            gpr,
            np.max(y_train),
            x_scaled,
            y_train,
            scaled_lower_bound,
            scaled_upper_bound,
            policy_num,
            config.horizon,
            movement_budget_remaining=movement_budget_remaining,
            movement_budget_total=rlbo_config.movement_budget,
        )
        decision_time = time.time() - start_t
        decision_times.append(decision_time)

        x_next_raw = x_scaler.inverse_transform_mean(x_next_scaled)
        y_next = test_func.real(x_next_raw)

        scaled_action_scale = np.maximum(scaled_upper_bound - scaled_lower_bound, 1e-8)
        raw_action_scale = np.maximum((upper_bound - lower_bound).reshape(-1), 1e-8)
        scaled_move_cost = float(
            np.linalg.norm((x_next_scaled.reshape(-1) - prev_x_scaled) / scaled_action_scale, ord=2)
        )
        raw_move_cost = float(
            np.linalg.norm((x_next_raw.reshape(-1) - prev_x_raw) / raw_action_scale, ord=2)
        )
        total_scaled_move_cost += scaled_move_cost
        total_raw_move_cost += raw_move_cost
        if movement_budget_remaining is not None:
            movement_budget_remaining = max(0.0, movement_budget_remaining - scaled_move_cost)

        x_train = np.vstack((x_train, x_next_raw))
        y_train = np.vstack((y_train, y_next))
        regret = float(0 - np.max(y_train))

        rows.append(
            {
                "Run": seed,
                "Iteration": iteration,
                "Regret": regret,
                "Scaled Move Cost": scaled_move_cost,
                "Raw Move Cost": raw_move_cost,
                "Cumulative Scaled Move Cost": total_scaled_move_cost,
                "Cumulative Raw Move Cost": total_raw_move_cost,
                "Decision Time": decision_time,
            }
        )

    return {
        "run_id": seed,
        "rows": rows,
        "total_scaled_move_cost": total_scaled_move_cost,
        "total_raw_move_cost": total_raw_move_cost,
        "final_regret": rows[-1]["Regret"] if rows else None,
        "avg_time": float(np.mean(decision_times)) if decision_times else 0.0,
    }


def main():
    args = parse_args()
    best_config_paths = [resolve_path(path) for path in args.best_config]
    if not best_config_paths:
        for reward in args.reward:
            best_config_paths.append(resolve_path(f"{args.results_root}/{reward}/tuning/best_config.json"))

    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_summaries = []

    for best_config_path in best_config_paths:
        summary = calculate_for_best_config(best_config_path, output_root, args)
        aggregate_summaries.append(summary)

    aggregate_path = output_root / "movement_cost_summary.json"
    aggregate_path.write_text(json.dumps(aggregate_summaries, indent=2, sort_keys=True))
    print(json.dumps(aggregate_summaries, indent=2, sort_keys=True))


def calculate_for_best_config(best_config_path, output_root, args):
    payload = json.loads(best_config_path.read_text())
    config, rlbo_config, ppo_config = apply_settings(payload, args.device, args.num_runs)
    reward_name = rlbo_config.reward_mode
    output_dir = output_root / reward_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    for run_id in range(config.num_runs):
        x_start = make_run_start(run_id, config)
        result = run_trial_with_budget(run_id, x_start, run_id + 1, config, rlbo_config, ppo_config)
        all_rows.extend(result["rows"])
        summaries.append(
            {
                "run_id": run_id,
                "total_scaled_move_cost": result["total_scaled_move_cost"],
                "total_raw_move_cost": result["total_raw_move_cost"],
                "final_regret": result["final_regret"],
                "avg_time": result["avg_time"],
            }
        )

    detail_csv = output_dir / "movement_cost_by_iteration.csv"
    with detail_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "best_config": str(best_config_path),
        "reward": reward_name,
        "num_runs": config.num_runs,
        "num_experiments": config.num_experiments,
        "snake_path_cost_weight": rlbo_config.snake_path_cost_weight,
        "movement_budget": rlbo_config.movement_budget,
        "reward_params": rlbo_config.reward_params,
        "mean_total_scaled_move_cost": float(np.mean([row["total_scaled_move_cost"] for row in summaries])),
        "std_total_scaled_move_cost": float(np.std([row["total_scaled_move_cost"] for row in summaries])),
        "mean_total_raw_move_cost": float(np.mean([row["total_raw_move_cost"] for row in summaries])),
        "std_total_raw_move_cost": float(np.std([row["total_raw_move_cost"] for row in summaries])),
        "runs": summaries,
        "detail_csv": str(detail_csv),
    }
    summary_json = output_dir / "movement_cost_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
