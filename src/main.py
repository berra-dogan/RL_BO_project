from rl_bo import RL_BO
from pure_bo import PureBO
from utils import Scaler
from objective_functions import ObjectiveFunctions
from config import ExperimentConfig, GPRConfig, PPOConfig, RLBOConfig
from experiments.used_budget import write_used_budget_summary
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
    def run_trial(
        seed,
        x_start,
        policy_num,
        config: ExperimentConfig,
        rlbo_config: RLBOConfig,
        ppo_config: PPOConfig,
        iterations_to_run,
        initial_state=None,
    ):
        try:
            # 1. Setup
            np.random.seed(seed)
            torch.manual_seed(seed)
            if config.device == "cuda":
                torch.cuda.manual_seed_all(seed)
            obj_funcs = ObjectiveFunctions(config.dimension)
            test_func = obj_funcs.functions[config.test_func_name] # Uses robust getter

            if initial_state is not None:
                x_train = initial_state["x_train"]
                y_train = initial_state["y_train"]
                regrets = list(initial_state["regrets"])
                decision_times = list(initial_state["decision_times"])
                scaled_move_costs = list(initial_state["scaled_move_costs"])
                raw_move_costs = list(initial_state["raw_move_costs"])
                movement_budget_remaining = initial_state["movement_budget_remaining"]
            else:
                x_train, y_train = x_start, test_func.real(x_start).reshape(-1, 1)
                regrets, decision_times = [], []
                scaled_move_costs, raw_move_costs = [], []
                movement_budget_remaining = rlbo_config.movement_budget

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
            if config.acquisition == "pure_bo":
                pure_bo_restarts = (
                    config.pure_bo_restarts
                    if config.pure_bo_restarts is not None
                    else 5 * config.dimension
                )
                acq_func = PureBO(
                    n_restarts=pure_bo_restarts,
                    xi=config.pure_bo_xi,
                    device=config.device,
                )
            else:
                acq_func = RL_BO(rlbo_config, ppo_config=ppo_config, device=config.device)

            lower_bound = np.full((1, config.dimension), config.lower_bound)
            upper_bound = np.full((1, config.dimension), config.upper_bound)

            # 3. Optimization Loop
            for _ in range(iterations_to_run):
                y_scaled = y_scaler.fit_transform(y_train)
                x_scaled = x_scaler.fit_transform(x_train)
                scaled_lower_bound = ((lower_bound - x_scaler.mu) / x_scaler.std).reshape(-1)
                scaled_upper_bound = ((upper_bound - x_scaler.mu) / x_scaler.std).reshape(-1)
                prev_x_scaled = x_scaled[-1].reshape(-1)
                prev_x_raw = x_train[-1].reshape(-1)

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
                    movement_budget_remaining=movement_budget_remaining,
                    movement_budget_total=rlbo_config.movement_budget,
                )
                decision_times.append(time.time() - start_t)
                action_scale = np.maximum(scaled_upper_bound - scaled_lower_bound, 1e-8)
                move_cost = np.linalg.norm((x_next_scaled.reshape(-1) - prev_x_scaled) / action_scale, ord=2)
                if movement_budget_remaining is not None:
                    movement_budget_remaining = max(0.0, movement_budget_remaining - float(move_cost))

                # Post-process and Update
                x_next = x_scaler.inverse_transform_mean(x_next_scaled)
                y_next = test_func.real(x_next)
                raw_action_scale = np.maximum((upper_bound - lower_bound).reshape(-1), 1e-8)
                raw_move_cost = np.linalg.norm((x_next.reshape(-1) - prev_x_raw) / raw_action_scale, ord=2)
                
                x_train = np.vstack((x_train, x_next))
                y_train = np.vstack((y_train, y_next))
                regrets.append(0 - np.max(y_train)) # Assuming global max is 0
                scaled_move_costs.append(float(move_cost))
                raw_move_costs.append(float(raw_move_cost))

            return {
                "run_id": seed,
                "regrets": regrets,
                "scaled_move_costs": scaled_move_costs,
                "raw_move_costs": raw_move_costs,
                "decision_times": decision_times,
                "avg_time": float(np.mean(decision_times)),
                "std_time": float(np.std(decision_times)),
                "x_train": x_train,
                "y_train": y_train,
                "movement_budget_remaining": movement_budget_remaining,
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
    parser.add_argument("--test-func", choices=("ackley", "sphere", "sum_square", "levy", "rosenbrock", "rastrigin", "schwefel", "michalewicz"), default=None)
    parser.add_argument("--num-experiments", type=int, default=None)
    parser.add_argument("--num-initial-data", type=int, default=None)
    parser.add_argument("--lower-bound", type=float, default=None)
    parser.add_argument("--upper-bound", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--acquisition", choices=("rl_bo", "pure_bo"), default=None)
    parser.add_argument("--pure-bo-restarts", type=int, default=None)
    parser.add_argument("--pure-bo-xi", type=float, default=None)
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
    parser.add_argument("--movement-budget", type=float, default=None)
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
    if args.acquisition is not None:
        config.acquisition = args.acquisition
    if args.pure_bo_restarts is not None:
        config.pure_bo_restarts = args.pure_bo_restarts
    if args.pure_bo_xi is not None:
        config.pure_bo_xi = args.pure_bo_xi

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
    if args.movement_budget is not None:
        rlbo_config.movement_budget = args.movement_budget
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


def checkpoint_paths(output_dir: Path, run_id: int):
    npz_path = output_dir / f"checkpoint_run_{run_id:04d}.npz"
    meta_path = output_dir / f"checkpoint_run_{run_id:04d}.json"
    return npz_path, meta_path


def checkpoint_fingerprint(config: ExperimentConfig, rlbo_config: RLBOConfig):
    """Config fields that must match between the original run and a resume."""
    return {
        "dimension": config.dimension,
        "test_func_name": config.test_func_name,
        "horizon": config.horizon,
        "lower_bound": config.lower_bound,
        "upper_bound": config.upper_bound,
        "num_initial_data": config.num_initial_data,
        "reward_mode": rlbo_config.reward_mode,
        "acquisition": config.acquisition,
    }


def save_checkpoint(output_dir: Path, config: ExperimentConfig, rlbo_config: RLBOConfig, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = checkpoint_paths(output_dir, result["run_id"])
    movement_budget_remaining = result.get("movement_budget_remaining")
    np.savez(
        npz_path,
        x_train=result["x_train"],
        y_train=result["y_train"],
        regrets=np.array(result["regrets"], dtype=float),
        scaled_move_costs=np.array(result["scaled_move_costs"], dtype=float),
        raw_move_costs=np.array(result["raw_move_costs"], dtype=float),
        decision_times=np.array(result["decision_times"], dtype=float),
        movement_budget_remaining=np.array(
            [movement_budget_remaining if movement_budget_remaining is not None else np.nan]
        ),
    )
    meta_path.write_text(json.dumps({
        "run_id": result["run_id"],
        "iterations_completed": len(result["regrets"]),
        "movement_budget_active": movement_budget_remaining is not None,
        "fingerprint": checkpoint_fingerprint(config, rlbo_config),
    }, indent=2, sort_keys=True) + "\n")


def load_checkpoint(output_dir: Path, run_id: int, config: ExperimentConfig, rlbo_config: RLBOConfig):
    npz_path, meta_path = checkpoint_paths(output_dir, run_id)
    if not npz_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    expected = checkpoint_fingerprint(config, rlbo_config)
    if meta["fingerprint"] != expected:
        raise ValueError(
            f"Checkpoint for run {run_id} in {output_dir} was made with a different config.\n"
            f"Checkpoint config: {meta['fingerprint']}\nRequested config: {expected}\n"
            "Refusing to resume from a mismatched checkpoint; use a different "
            "--output-dir or delete the stale checkpoint files."
        )

    data = np.load(npz_path)
    movement_budget_remaining = (
        float(data["movement_budget_remaining"][0]) if meta["movement_budget_active"] else None
    )
    return {
        "x_train": data["x_train"],
        "y_train": data["y_train"],
        "regrets": list(data["regrets"]),
        "scaled_move_costs": list(data["scaled_move_costs"]),
        "raw_move_costs": list(data["raw_move_costs"]),
        "decision_times": list(data["decision_times"]),
        "movement_budget_remaining": movement_budget_remaining,
        "iterations_completed": meta["iterations_completed"],
    }


def write_run_result(output_dir: Path, config: ExperimentConfig, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / f"run_{result['run_id']:04d}.csv"
    fieldnames = [
        "Iteration",
        "Regret",
        "Scaled Move Cost",
        "Raw Move Cost",
        "Cumulative Scaled Move Cost",
        "Cumulative Raw Move Cost",
        "Avg Time",
        "Std Time",
    ]
    with run_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        cumulative_scaled_move_cost = 0.0
        cumulative_raw_move_cost = 0.0
        for iteration, regret in enumerate(result["regrets"]):
            scaled_move_cost = result.get("scaled_move_costs", [0.0] * len(result["regrets"]))[iteration]
            raw_move_cost = result.get("raw_move_costs", [0.0] * len(result["regrets"]))[iteration]
            cumulative_scaled_move_cost += scaled_move_cost
            cumulative_raw_move_cost += raw_move_cost
            writer.writerow({
                "Iteration": iteration,
                "Regret": regret,
                "Scaled Move Cost": scaled_move_cost,
                "Raw Move Cost": raw_move_cost,
                "Cumulative Scaled Move Cost": cumulative_scaled_move_cost,
                "Cumulative Raw Move Cost": cumulative_raw_move_cost,
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
    scaled_move_costs = np.array([
        [float(row.get("Scaled Move Cost", 0.0)) for row in rows]
        for rows in run_rows
    ])
    raw_move_costs = np.array([
        [float(row.get("Raw Move Cost", 0.0)) for row in rows]
        for rows in run_rows
    ])
    cumulative_scaled_move_costs = np.cumsum(scaled_move_costs, axis=1)
    cumulative_raw_move_costs = np.cumsum(raw_move_costs, axis=1)
    avg_time = np.mean([float(rows[0]["Avg Time"]) for rows in run_rows])
    std_time = np.mean([float(rows[0]["Std Time"]) for rows in run_rows])

    summary_path = output_dir / f"RL_BO_{config.dimension}D_{config.test_func_name}_h{config.horizon}.csv"
    fieldnames = [
        "Avg Regret",
        "Std Regret",
        "Avg Scaled Move Cost",
        "Std Scaled Move Cost",
        "Avg Raw Move Cost",
        "Std Raw Move Cost",
        "Avg Cumulative Scaled Move Cost",
        "Std Cumulative Scaled Move Cost",
        "Avg Cumulative Raw Move Cost",
        "Std Cumulative Raw Move Cost",
        "Avg Time",
        "Std Time",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for iteration in range(regrets.shape[1]):
            writer.writerow({
                "Avg Regret": np.mean(regrets[:, iteration]),
                "Std Regret": np.std(regrets[:, iteration]),
                "Avg Scaled Move Cost": np.mean(scaled_move_costs[:, iteration]),
                "Std Scaled Move Cost": np.std(scaled_move_costs[:, iteration]),
                "Avg Raw Move Cost": np.mean(raw_move_costs[:, iteration]),
                "Std Raw Move Cost": np.std(raw_move_costs[:, iteration]),
                "Avg Cumulative Scaled Move Cost": np.mean(
                    cumulative_scaled_move_costs[:, iteration]
                ),
                "Std Cumulative Scaled Move Cost": np.std(
                    cumulative_scaled_move_costs[:, iteration]
                ),
                "Avg Cumulative Raw Move Cost": np.mean(
                    cumulative_raw_move_costs[:, iteration]
                ),
                "Std Cumulative Raw Move Cost": np.std(
                    cumulative_raw_move_costs[:, iteration]
                ),
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
        write_used_budget_summary(output_dir, rlbo_config.movement_budget)
        return

    if args.run_id is not None:
        run_ids = [args.run_id]
    else:
        run_ids = list(range(config.num_runs))

    results = []
    skipped = 0
    for run_id in run_ids:
        checkpoint = load_checkpoint(output_dir, run_id, config, rlbo_config)
        already_completed = checkpoint["iterations_completed"] if checkpoint else 0

        if already_completed >= config.num_experiments:
            skipped += 1
            run_csv = output_dir / f"run_{run_id:04d}.csv"
            if run_csv.exists():
                print(
                    f"Run {run_id} already has {already_completed} iterations "
                    f"(requested {config.num_experiments}); skipping."
                )
                continue
            # The checkpoint says this run is done, but its CSV is missing -
            # e.g. the original run was killed after checkpointing but before
            # the CSV-writing pass. Regenerate the CSV from the checkpoint
            # instead of silently leaving it missing forever.
            print(
                f"Run {run_id} has a complete checkpoint but no CSV; "
                "regenerating its CSV from the checkpoint instead of re-running."
            )
            n = config.num_experiments
            decision_times = checkpoint["decision_times"][:n]
            results.append({
                "run_id": run_id,
                "regrets": checkpoint["regrets"][:n],
                "scaled_move_costs": checkpoint["scaled_move_costs"][:n],
                "raw_move_costs": checkpoint["raw_move_costs"][:n],
                "avg_time": float(np.mean(decision_times)) if decision_times else 0.0,
                "std_time": float(np.std(decision_times)) if decision_times else 0.0,
            })
            continue

        if checkpoint:
            print(f"Resuming run {run_id} from {already_completed} to {config.num_experiments} iterations.")
            x_start = None
        else:
            x_start = make_run_start(run_id, config)

        result = BOEngine.run_trial(
            run_id,
            x_start,
            run_id + 1,
            config,
            rlbo_config,
            ppo_config,
            iterations_to_run=config.num_experiments - already_completed,
            initial_state=checkpoint,
        )
        if result is not None:
            results.append(result)
            save_checkpoint(output_dir, config, rlbo_config, result)

    if not results and not skipped:
        print("all trials failed.")
        return

    for result in results:
        run_path = write_run_result(output_dir, config, result)
        print(f"Run {result['run_id']} saved to {run_path}")

    if args.run_id is None:
        aggregate_results(output_dir, config)
    write_used_budget_summary(output_dir, rlbo_config.movement_budget)

if __name__ == '__main__':
    main()
