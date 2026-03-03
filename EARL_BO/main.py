from rl_bo import RL_BO
from utils import Scaler
from objective_functions import ObjectiveFunctions
from config import ExperimentConfig, RLBOConfig

import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern


class BOEngine:
    """Handles the execution of a single Bayesian Optimization trial."""
    
    @staticmethod
    def run_trial(seed, x_start, policy_num, config: ExperimentConfig):
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
            acq_func = RL_BO(RLBOConfig(), device=config.device)
            
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
    return parser.parse_args()


def make_run_start(run_id, config: ExperimentConfig):
    rng = np.random.default_rng(run_id)
    return rng.uniform(config.lower_bound, config.upper_bound, (config.num_initial_data, config.dimension))


def write_run_result(output_dir: Path, config: ExperimentConfig, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / f"run_{result['run_id']:04d}.csv"
    df = pd.DataFrame({
        "Iteration": np.arange(len(result["regrets"])),
        "Regret": result["regrets"],
        "Avg Time": result["avg_time"],
        "Std Time": result["std_time"],
    })
    df.to_csv(run_path, index=False)
    return run_path


def aggregate_results(output_dir: Path, config: ExperimentConfig):
    run_files = sorted(output_dir.glob("run_*.csv"))
    if not run_files:
        print(f"No per-run results found in {output_dir}")
        return None

    run_frames = [pd.read_csv(path) for path in run_files]
    regrets = np.array([frame["Regret"].to_numpy() for frame in run_frames])
    df = pd.DataFrame({
        "Avg Regret": np.mean(regrets, axis=0),
        "Std Regret": np.std(regrets, axis=0),
        "Avg Time": np.mean([frame["Avg Time"].iloc[0] for frame in run_frames]),
        "Std Time": np.mean([frame["Std Time"].iloc[0] for frame in run_frames]),
    })

    summary_path = output_dir / f"RL_BO_{config.dimension}D_{config.test_func_name}_h{config.horizon}.csv"
    df.to_csv(summary_path, index=False)
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
    config = ExperimentConfig(device=resolve_device(args.device))
    if args.num_runs is not None:
        config.num_runs = args.num_runs
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
        result = BOEngine.run_trial(run_id, x_start, run_id + 1, config)
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
