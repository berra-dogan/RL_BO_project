import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from utils import Scaler
from rl_bo import RL_BO
from config import RLBOConfig
import time


class BayesianOptimizer:
    """Handles the execution of a single BO trajectory."""
    
    def __init__(self, config, objective_lib):
        self.cfg = config
        self.obj_lib = objective_lib

    def run(self, seed, x_init, policy_num, horizon):
        """Executes one BO trajectory."""
        np.random.seed(seed)
        
        # State initialization
        x_train, y_train = x_init, self.obj_lib.evaluate(self.cfg.test_func_name, x_init).reshape(-1, 1)
        
        # Model setup
        gpr = GaussianProcessRegressor(kernel=RBF(1.0) + WhiteKernel(noise_level=1e-05), n_restarts_optimizer=10)
        x_scaler, y_scaler = Scaler(), Scaler()
        acq_func = RL_BO(RLBOConfig())
        
        results = {'regrets': [], 'times': []}
        bounds = (np.full(self.cfg.dimension, self.cfg.lower_bound), 
                  np.full(self.cfg.dimension, self.cfg.upper_bound))

        for _ in range(self.cfg.num_experiments):
            # Fit and Transform
            y_scaled = y_scaler.fit_transform(y_train)
            x_scaled = x_scaler.fit_transform(x_train)
            gpr.fit(x_scaled, y_scaled)

            # Decision making with timing
            start = time.time()
            x_next_scaled = acq_func.evaluate(gpr, np.max(y_scaled), x_scaled, y_scaled, 
                                             bounds[0], bounds[1], policy_num, horizon)
            results['times'].append(time.time() - start)

            # Update dataset
            x_next = x_scaler.inverse_transform_mean(x_next_scaled)
            y_next = self.obj_lib.evaluate(self.cfg.test_func_name, x_next)
            
            x_train = np.vstack([x_train, x_next])
            y_train = np.vstack([y_train, y_next])
            results['regrets'].append(0 - np.max(y_train)) # Assuming y_max_real = 0

        return results
