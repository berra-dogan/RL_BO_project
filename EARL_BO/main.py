import numpy as np
import torch
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from functools import partial
import pandas as pd
import multiprocessing as mp
from dataclasses import dataclass

from rl_bo import RL_BO
from utils import Scaler
from objective_functions import ObjectiveFunctions


def run_single_experiment(seed, X_start_candidate, policy_file_num, test_func_name, dimension, num_experiments,
                          num_initial_data, lower_bound, upper_bound, horizon):
    try:
        import time  # Add time import for tracking decision times

        np.random.seed(seed)
        torch.manual_seed(seed)
        obj_funcs = ObjectiveFunctions(dimension)
        test_func = obj_funcs.functions[test_func_name]
        y_max_real = 0
        GP_lengthscale = 1

        regrets = []
        X_train = X_start_candidate
        y_train = test_func.real(X_train).reshape(-1, 1)
        y_max = np.max(y_train)
        print(-y_max)
        regret = y_max_real - y_max
        regrets.append(regret)

        action_min = np.array([lower_bound] * dimension)
        action_max = np.array([upper_bound] * dimension)

        # Surrogate model
        kernel = RBF(GP_lengthscale, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        scaler = Scaler()
        acq_func = RL_BO()

        # Create a list to store decision times
        decision_times = []

        for j in range(num_experiments):
            acq_func.horizon = horizon
            scaler.fit(y_train)
            y_train_scaled = scaler.transform(y_train)
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            gpr.fit(X_train_scaled, y_train_scaled)
            y_max_scaled = np.max(y_train_scaled)

            # Start timing the decision process
            start_time = time.time()
            print(f"\n===== Running iteration {j + 1}/{num_experiments} for seed {seed} =====")

            X_next = acq_func.evaluate(gpr, y_max_scaled, X_train_scaled, y_train_scaled, action_min, action_max,
                                       policy_file_num, horizon)

            # End timing and store the elapsed time
            elapsed_time = time.time() - start_time
            decision_times.append(elapsed_time)
            print(f"Decision {j + 1}/{num_experiments} took {elapsed_time:.4f} seconds")

            X_next = scaler.inverse_transform_mean(X_next)
            y_next = test_func.real(X_next)
            X_train = np.vstack((X_train, X_next))
            y_train = np.vstack((y_train, y_next))

            y_max = np.max(y_train)
            regret = y_max_real - y_max
            regrets.append(regret)

        # Calculate timing statistics
        avg_decision_time = np.mean(decision_times)
        std_decision_time = np.std(decision_times)
        print(f"Average decision time: {avg_decision_time:.4f} seconds, Std: {std_decision_time:.4f} seconds")

        # Return timing statistics along with regrets
        return regrets, avg_decision_time, std_decision_time

    except Exception as e:
        print(f"Error in experiment with seed {seed}: {str(e)}")
        return None

@dataclass
class ExperimentConfig:
    """Container for experiment hyperparameters."""
    dimension = 30
    test_func_name = 'ackley' #'Ackley', 'Sum_square', 'Levy', 'Rosenbrock'
    num_runs = 1
    horizons = [3]  # List of horizons to run
    num_experiments = 1
    num_initial_data = 30
    lower_bound = -2 #normalized bound
    upper_bound = 2  #normalized bound
    num_workers = 10  # Fixed number of workers


if __name__ == '__main__':
    config = ExperimentConfig()
    # Run for each horizon in the list
    for horizon in config.horizons:
        print(f"Starting experiments with horizon = {horizon}")

        np.random.seed(1)
        X_start_candidates = [np.random.uniform(low=-15, high=15, size=(config.num_initial_data, config.dimension)) for _ in
                              range(config.num_runs)]

        # Prepare the partial function with fixed arguments
        partial_run_experiment = partial(
            run_single_experiment,
            test_func_name=config.test_func_name,
            dimension=config.dimension,
            num_experiments=config.num_experiments,
            num_initial_data=config.num_initial_data,
            lower_bound=config.lower_bound,
            upper_bound=config.upper_bound,
            horizon=horizon
        )

        # Create a pool of workers with fixed number
        with mp.Pool(processes=config.num_workers) as pool:
            # Run the experiments in parallel
            results = pool.starmap(partial_run_experiment,
                                   [(i, X_start_candidates[i], i + 1)
                                    for i in range(config.num_runs)])

        # Filter out None results (from failed experiments)
        results = [r for r in results if r is not None]

        # Separate regrets and timing data
        regret_store = np.array([r[0] for r in results])
        decision_times_avg = np.array([r[1] for r in results])
        decision_times_std = np.array([r[2] for r in results])

        # Process the regret results
        avg_regret = np.mean(regret_store, axis=0)
        avg_std = np.std(regret_store, axis=0)

        # Calculate overall timing statistics
        overall_avg_decision_time = np.mean(decision_times_avg)
        overall_std_decision_time = np.mean(decision_times_std)

        # Create DataFrame with both regret and timing data
        df = pd.DataFrame({
            'Avg Regret': avg_regret,
            'Avg Std': avg_std,
            'Avg Decision Time': overall_avg_decision_time,
            'Std Decision Time': overall_std_decision_time
        })

        filename = f'RL_BO_{config.dimension}D_{config.test_func_name}_h{horizon}.csv'
        df.to_csv(filename, index=False)
        print(f"Experiment completed for horizon {horizon}. Results saved to '{filename}'.")
        print(f"Average decision time: {overall_avg_decision_time:.4f}s, Std: {overall_std_decision_time:.4f}s")

    print("All experiments for all horizons completed.")