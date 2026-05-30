from rl_bo import RL_BO
from utils import Scaler
from objective_functions import ObjectiveFunctions
from config import ExperimentConfig, GPRConfig, PPOConfig, RLBOConfig
from rewards import available_reward_modes

import argparse
from pathlib import Path

import numpy as np
import torch
import time
import csv
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern


class BOEngine:
    """Handles the execution of a single Bayesian Optimization trial."""
    
    @staticmethod
    def run_trial(seed, x_start, policy_num, config: ExperimentConfig, rlbo_config: RLBOConfig, ppo_config: PPOConfig):
        try:
            # 1. Setup
            np.random.seed(seed)
            torch.manual_seed(seed)
            if config.device == "cuda":
                torch.cuda.manual_seed_all(seed)
            obj_funcs = ObjectiveFunctions(config.dimension)
            test_func = obj_funcs.functions[config.test_func_name] # Uses robust getter
            
            x_train, y_train = x_start, test_func.real(x_start).reshape(-1, 1)
            
            # 2. Models
            gpr_config = config.gpr_config
            kernel = RBF(length_scale=gpr_config.rbf_length_scale, length_scale_bounds=gpr_config.rbf_length_scale_bounds) \
                    + WhiteKernel(noise_level=gpr_config.wk_noise_level, noise_level_bounds=gpr_config.wk_noise_level_bounds)

            # kernel = Matern(length_scale=gpr_config.rbf_length_scale, length_scale_bounds=gpr_config.rbf_length_scale_bounds) \
            #         + WhiteKernel(noise_level=gpr_config.wk_noise_level, noise_level_bounds=gpr_config.wk_noise_level_bounds)

            gpr = GaussianProcessRegressor(
                kernel=kernel, 
                alpha=gpr_config.alpha,  # Small jitter for stability
                n_restarts_optimizer=gpr_config.n_restarts_optimizer
            )
            x_scaler, y_scaler = Scaler(), Scaler()
            acq_func = RL_BO(rlbo_config, ppo_config=ppo_config, device=config.device)
            
            regrets, decision_times = [], []
            lower_bound = np.full((1, config.dimension), config.lower_bound)
            upper_bound = np.full((1, config.dimension), config.upper_bound)

            # 3. Optimization Loop
            for _ in range(config.num_experiments):
                y_scaled = y_scaler.fit_transform(y_train)
                x_scaled = x_scaler.fit_transform(x_train)
                scaled_lower_bound = ((lower_bound - x_scaler.mu) / x_scaler.std).reshape(-1)
                scaled_upper_bound = ((upper_bound - x_scaler.mu) / x_scaler.std).reshape(-1)

                # Step
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
                )
                decision_times.append(time.time() - start_t)

                # Post-process and Update
                x_next = x_scaler.inverse_transform_mean(x_next_scaled)
                y_next = test_func.real(x_next)
                
                x_train = np.vstack((x_train, x_next))
                y_train = np.vstack((y_train, y_next))
                regrets.append(0 - np.max(y_train)) # Assuming global max is 0

            return {
                "run_id": seed,
                "regrets": regrets,
                "avg_time": float(np.mean(decision_times)),
                "std_time": float(np.std(decision_times)),
            }

        except Exception as e:
            print(f"Trial {seed} failed: {e}")
            return None

def parse_args():
    parser = argparse.ArgumentParser(description="Run EARL BO experiments.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--run-id", type=int, default=None, help="Run a single trial id for cluster job arrays.")
    parser.add_argument("--output-dir", default="results", help="Directory for per-run CSVs and summary output.")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip execution and aggregate existing per-run results.")
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--test-func", choices=("ackley", "sphere", "sum_square", "levy", "rosenbrock"), default=None)
    parser.add_argument("--num-experiments", type=int, default=None)
    parser.add_argument("--num-initial-data", type=int, default=None)
    parser.add_argument("--lower-bound", type=float, default=None)
    parser.add_argument("--upper-bound", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--gpr-rbf-length-scale", type=float, default=None)
    parser.add_argument("--gpr-rbf-length-scale-bounds", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--gpr-wk-noise-level", type=float, default=None)
    parser.add_argument("--gpr-wk-noise-level-bounds", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--gpr-alpha", type=float, default=None)
    parser.add_argument("--gpr-restarts", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--update-episode", type=int, default=None)
    parser.add_argument("--off-policy-episodes", type=int, default=None)
    parser.add_argument("--no-improvement-threshold", type=int, default=None)
    parser.add_argument("--encoder-learning-rate", type=float, default=None)
    parser.add_argument("--reward-mode", choices=available_reward_modes(), default=None)
    parser.add_argument("--snake-path-cost-weight", type=float, default=None)
    parser.add_argument(
        "--reward-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Reward-specific numeric parameter. Can be repeated.",
    )
    parser.add_argument(
        "--reward-params-json",
        default=None,
        help="JSON object of reward-specific numeric parameters.",
    )
    parser.add_argument("--ppo-learning-rate", type=float, default=None)
    parser.add_argument("--ppo-action-std", type=float, default=None)
    parser.add_argument("--ppo-action-std-min", type=float, default=None)
    parser.add_argument("--ppo-action-decay", type=float, default=None)
    parser.add_argument("--ppo-k-epochs", type=int, default=None)
    parser.add_argument("--ppo-eps-clip", type=float, default=None)
    parser.add_argument("--ppo-gamma", type=float, default=None)
    parser.add_argument("--ppo-gamma-increase", type=float, default=None)
    parser.add_argument("--ppo-vf-coeff", type=float, default=None)
    parser.add_argument("--ppo-entropy-coeff", type=float, default=None)
    parser.add_argument("--ppo-freeze-num", type=int, default=None)
    return parser.parse_args()


def build_configs(args):
    config = ExperimentConfig(device=resolve_device(args.device))
    if args.num_runs is not None:
        config.num_runs = args.num_runs
    if args.dimension is not None:
        config.dimension = args.dimension
    if args.test_func is not None:
        config.test_func_name = args.test_func
    if args.num_experiments is not None:
        config.num_experiments = args.num_experiments
    if args.num_initial_data is not None:
        config.num_initial_data = args.num_initial_data
    if args.lower_bound is not None:
        config.lower_bound = args.lower_bound
    if args.upper_bound is not None:
        config.upper_bound = args.upper_bound
    if args.horizon is not None:
        config.horizon = args.horizon

    gpr_config = GPRConfig()
    if args.gpr_rbf_length_scale is not None:
        gpr_config.rbf_length_scale = args.gpr_rbf_length_scale
    if args.gpr_rbf_length_scale_bounds is not None:
        gpr_config.rbf_length_scale_bounds = tuple(args.gpr_rbf_length_scale_bounds)
    if args.gpr_wk_noise_level is not None:
        gpr_config.wk_noise_level = args.gpr_wk_noise_level
    if args.gpr_wk_noise_level_bounds is not None:
        gpr_config.wk_noise_level_bounds = tuple(args.gpr_wk_noise_level_bounds)
    if args.gpr_alpha is not None:
        gpr_config.alpha = args.gpr_alpha
    if args.gpr_restarts is not None:
        gpr_config.n_restarts_optimizer = args.gpr_restarts
    config.gpr_config = gpr_config

    rlbo_config = RLBOConfig()
    if args.max_episodes is not None:
        rlbo_config.max_episodes = args.max_episodes
    if args.update_episode is not None:
        rlbo_config.update_episode = args.update_episode
    if args.off_policy_episodes is not None:
        rlbo_config.off_policy_episodes = args.off_policy_episodes
    if args.no_improvement_threshold is not None:
        rlbo_config.no_improvement_threshold = args.no_improvement_threshold
    if args.encoder_learning_rate is not None:
        rlbo_config.encoder_learning_rate = args.encoder_learning_rate
    if args.reward_mode is not None:
        rlbo_config.reward_mode = args.reward_mode
    if args.snake_path_cost_weight is not None:
        rlbo_config.snake_path_cost_weight = args.snake_path_cost_weight
    rlbo_config.reward_params = parse_reward_params(args)

    ppo_config = PPOConfig()
    if args.ppo_learning_rate is not None:
        ppo_config.learning_rate = args.ppo_learning_rate
    if args.ppo_action_std is not None:
        ppo_config.action_std = args.ppo_action_std
    if args.ppo_action_std_min is not None:
        ppo_config.action_std_min = args.ppo_action_std_min
    if args.ppo_action_decay is not None:
        ppo_config.action_decay = args.ppo_action_decay
    if args.ppo_k_epochs is not None:
        ppo_config.K_epochs = args.ppo_k_epochs
    if args.ppo_eps_clip is not None:
        ppo_config.eps_clip = args.ppo_eps_clip
    if args.ppo_gamma is not None:
        ppo_config.gamma = args.ppo_gamma
    if args.ppo_gamma_increase is not None:
        ppo_config.gamma_increase = args.ppo_gamma_increase
    if args.ppo_vf_coeff is not None:
        ppo_config.VF_coeff = args.ppo_vf_coeff
    if args.ppo_entropy_coeff is not None:
        ppo_config.Entropy_coeff = args.ppo_entropy_coeff
    if args.ppo_freeze_num is not None:
        ppo_config.freeze_num = args.ppo_freeze_num

    return config, rlbo_config, ppo_config


def parse_reward_params(args):
    params = {}
    if args.reward_params_json is not None:
        decoded = json.loads(args.reward_params_json)
        if not isinstance(decoded, dict):
            raise ValueError("--reward-params-json must decode to an object")
        params.update({key: float(value) for key, value in decoded.items()})

    for item in args.reward_param:
        if "=" not in item:
            raise ValueError(f"Invalid --reward-param '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        params[key] = float(value)

    return params


def make_run_start(run_id, config: ExperimentConfig):
    rng = np.random.default_rng(run_id)
    return rng.uniform(config.lower_bound, config.upper_bound, (config.num_initial_data, config.dimension))


def write_run_result(output_dir: Path, config: ExperimentConfig, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / f"run_{result['run_id']:04d}.csv"
    with run_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Iteration", "Regret", "Avg Time", "Std Time"])
        writer.writeheader()
        for iteration, regret in enumerate(result["regrets"]):
            writer.writerow({
                "Iteration": iteration,
                "Regret": regret,
                "Avg Time": result["avg_time"],
                "Std Time": result["std_time"],
            })
    return run_path


def aggregate_results(output_dir: Path, config: ExperimentConfig):
    run_files = sorted(output_dir.glob("run_*.csv"))
    if not run_files:
        print(f"No per-run results found in {output_dir}")
        return None

    run_rows = []
    for path in run_files:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        run_rows.append(rows)

    regrets = np.array([[float(row["Regret"]) for row in rows] for rows in run_rows])
    avg_time = np.mean([float(rows[0]["Avg Time"]) for rows in run_rows])
    std_time = np.mean([float(rows[0]["Std Time"]) for rows in run_rows])

    summary_path = output_dir / f"RL_BO_{config.dimension}D_{config.test_func_name}_h{config.horizon}.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Avg Regret", "Std Regret", "Avg Time", "Std Time"])
        writer.writeheader()
        for avg_regret, std_regret in zip(np.mean(regrets, axis=0), np.std(regrets, axis=0)):
            writer.writerow({
                "Avg Regret": avg_regret,
                "Std Regret": std_regret,
                "Avg Time": avg_time,
                "Std Time": std_time,
            })
    print(f"Aggregated results saved to {summary_path}")
    return summary_path


def resolve_device(requested_device):
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False")
    return requested_device


def main():
    args = parse_args()
    config, rlbo_config, ppo_config = build_configs(args)
    output_dir = Path(args.output_dir)

    print(f"Starting {config.dimension}D {config.test_func_name} experiments on {config.device}...")
    if config.device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    if args.aggregate_only:
        aggregate_results(output_dir, config)
        return

    if args.run_id is not None:
        run_ids = [args.run_id]
    else:
        run_ids = list(range(config.num_runs))

    results = []
    for run_id in run_ids:
        x_start = make_run_start(run_id, config)
        result = BOEngine.run_trial(run_id, x_start, run_id + 1, config, rlbo_config, ppo_config)
        if result is not None:
            results.append(result)

    if not results:
        print("all trials failed.")
        return

    for result in results:
        run_path = write_run_result(output_dir, config, result)
        print(f"Run {result['run_id']} saved to {run_path}")

    if args.run_id is None:
        aggregate_results(output_dir, config)

if __name__ == '__main__':
    main()
