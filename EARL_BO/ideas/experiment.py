from dataclasses import dataclass
import multiprocessing as mp
import numpy as np
import pandas as pd

from objective_functions import ObjectiveFunctions
from ideas.bo import BayesianOptimizer

@dataclass
class ExperimentConfig:
    """Container for experiment hyperparameters."""
    dimension: int = 30
    test_func_name: str = 'ackley'
    num_runs: int = 1
    num_experiments: int = 50
    num_initial_data: int = 30
    lower_bound: float = -2.0
    upper_bound: float = 2.0
    num_workers: int = 10
    horizon: int = 2


class ExperimentRunner:
    """Orchestrates parallel execution and data persistence."""
    
    def __init__(self, config):
        self.cfg = config
        self.obj_lib = ObjectiveFunctions(config.dimension)
        self.opt = BayesianOptimizer(config, self.obj_lib)

    def run_all(self):
        horizon = self.cfg.horizon
        print(f"Running Horizon {horizon}...")
        np.random.seed(1) # Ensure same starting points for different horizons
        
        seeds = range(self.cfg.num_runs)
        x_starts = [np.random.uniform(-15, 15, (self.cfg.num_initial_data, self.cfg.dimension)) for _ in seeds]
        
        # Parallel Execution
        with mp.Pool(self.cfg.num_workers) as pool:
            raw_results = pool.starmap(self.opt.run, [(s, x_starts[s], s+1, horizon) for s in seeds])
        
        self.save_results(raw_results, horizon)

    def save_results(self, results: list[dict], horizon: int):
        # 1. Filter out None results to prevent crashes
        results = [r for r in results if r is not None]
        if not results: 
            print("No results to save.")
            return

        # 2. Extract data (Ensure keys match what BayesianOptimizer returns)
        # Assuming your optimizer returns 'times' as a list per iteration
        regret_matrix = np.array([r['regrets'] for r in results])
        
        # We calculate the mean of the times for each run first
        run_averages = [np.mean(r['times']) for r in results]
        run_stds = [np.std(r['times']) for r in results]

        # 3. Aggregate for the CSV
        # Each row in the CSV represents one step in the BO process (e.g., 50 rows)
        avg_regret = np.mean(regret_matrix, axis=0)
        std_regret = np.std(regret_matrix, axis=0)

        df = pd.DataFrame({
            'Avg Regret': avg_regret,
            'Std Regret': std_regret,
            # These will be scalar values repeated for every row
            'Overall_Avg_Time': np.mean(run_averages),
            'Overall_Std_Time': np.mean(run_stds)
        })

        # 4. Fix the naming: self.cfg instead of self.config
        fname = f"RL_BO_{self.cfg.dimension}D_{self.cfg.test_func_name}_h{horizon}.csv"
        df.to_csv(fname, index=False)
        print(f"Saved: {fname}")
