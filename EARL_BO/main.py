from rl_bo import RL_BO
from utils import Scaler
from objective_functions import ObjectiveFunctions
from config import ExperimentConfig, RLBOConfig

import numpy as np
import torch
import pandas as pd
import multiprocessing as mp
import time
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class BOEngine:
    """Handles the execution of a single Bayesian Optimization trial."""
    
    @staticmethod
    def run_trial(seed, x_start, policy_num, config: ExperimentConfig):
        try:
            # 1. Setup
            np.random.seed(seed)
            torch.manual_seed(seed)
            obj_funcs = ObjectiveFunctions(config.dimension)
            test_func = obj_funcs.functions[config.test_func_name] # Uses robust getter
            
            x_train, y_train = x_start, test_func.real(x_start).reshape(-1, 1)
            
            # 2. Models
            gpr_config = config.gpr_config
            kernel = RBF(length_scale=gpr_config.rbf_length_scale, length_scale_bounds=gpr_config.rbf_length_scale_bounds) \
                    + WhiteKernel(noise_level=gpr_config.wk_noise_level, noise_level_bounds=gpr_config.wk_noise_level_bounds)
            gpr = GaussianProcessRegressor(
                kernel=kernel, 
                alpha=gpr_config.alpha,  # Small jitter for stability
                n_restarts_optimizer=gpr_config.n_restarts_optimizer
            )
            scaler, acq_func = Scaler(), RL_BO(RLBOConfig())
            
            regrets, decision_times = [], []
            bounds = (np.full(config.dimension, config.lower_bound), 
                      np.full(config.dimension, config.upper_bound))

            # 3. Optimization Loop
            for _ in range(config.num_experiments):
                acq_func.horizon = config.horizon
                
                # Pre-process
                y_scaled = scaler.fit_transform(y_train)
                x_scaled = scaler.fit_transform(x_train)
                gpr.fit(x_scaled, y_scaled)

                # Step
                start_t = time.time()
                x_next_scaled = acq_func.evaluate(
                    gpr, np.max(y_scaled), x_scaled, y_scaled, 
                    bounds[0], bounds[1], policy_num, config.horizon
                )
                decision_times.append(time.time() - start_t)

                # Post-process and Update
                x_next = scaler.inverse_transform_mean(x_next_scaled)
                y_next = test_func.real(x_next)
                
                x_train = np.vstack((x_train, x_next))
                y_train = np.vstack((y_train, y_next))
                regrets.append(0 - np.max(y_train)) # Assuming global max is 0

            return regrets, np.mean(decision_times), np.std(decision_times)

        except Exception as e:
            print(f"Trial {seed} failed: {e}")
            return None

def main():
    # 1. Initialize Configuration
    config = ExperimentConfig()
    print(f"Starting {config.dimension}D {config.test_func_name} experiments...")

    # 2. Prepare Starting Points
    x_starts = [
        np.random.uniform(-15, 15, (config.num_initial_data, config.dimension)) 
        for _ in range(config.num_runs)
    ]

    # 3. Parallel Execution
    # partial fixes the config argument so starmap only needs trial-specific data
    worker_fn = partial(BOEngine.run_trial, config=config)
    tasks = [(i, x_starts[i], i + 1) for i in range(config.num_runs)]

    with mp.Pool(processes=config.num_workers) as pool:
        results = pool.starmap(worker_fn, tasks)

    # 4. Filter and Aggregate Data
    results = [r for r in results if r is not None]
    if not results:
        return

    regrets = np.array([r[0] for r in results])
    df = pd.DataFrame({
        'Avg Regret': np.mean(regrets, axis=0),
        'Std Regret': np.std(regrets, axis=0),
        'Avg Time': np.mean([r[1] for r in results]),
        'Std Time': np.mean([r[2] for r in results])
    })

    # 5. Save
    filename = f'RL_BO_{config.dimension}D_{config.test_func_name}_h{config.horizon}.csv'
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    main()