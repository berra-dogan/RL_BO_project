import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from utils import Scaler


class PureBO:
    """Classical GP + Expected Improvement Bayesian optimization baseline.

    No RL policy and no trust region: fits the same GPR used by RL_BO on
    all observed data, then picks the point that analytically maximizes
    Expected Improvement over the whole scaled action box via multi-start
    L-BFGS-B. Matches RL_BO.evaluate's signature so BOEngine.run_trial can
    swap the two acquisition functions without touching the rest of the
    pipeline (checkpointing, CSV output, move-cost/budget accounting).
    """

    def __init__(self, n_restarts=10, xi=0.01, device="cpu"):
        self.n_restarts = n_restarts
        self.xi = xi
        self.device = device
        self.scaler_y = Scaler()

    def evaluate(
        self,
        model,
        y_max_raw,
        X_train_scaled,
        y_train_raw,
        action_min_scaled,
        action_max_scaled,
        policy_file_num,
        horizon,
        movement_budget_remaining=None,
        movement_budget_total=None,
    ):
        y_train_scaled = self.scaler_y.fit_transform(y_train_raw)
        model.fit(X_train_scaled, y_train_scaled.ravel())
        y_best_scaled = float(np.max(y_train_scaled))

        action_min_scaled = np.asarray(action_min_scaled).reshape(-1)
        action_max_scaled = np.asarray(action_max_scaled).reshape(-1)
        bounds = list(zip(action_min_scaled, action_max_scaled))
        dim = X_train_scaled.shape[1]

        def negative_expected_improvement(x):
            mean, std = model.predict(x.reshape(1, -1), return_std=True)
            mean, std = float(mean[0]), float(std[0])
            if std < 1e-9:
                return 0.0
            z = (mean - y_best_scaled - self.xi) / std
            ei = (mean - y_best_scaled - self.xi) * norm.cdf(z) + std * norm.pdf(z)
            return -ei

        rng = np.random.default_rng(policy_file_num)
        starts = rng.uniform(
            action_min_scaled, action_max_scaled, size=(self.n_restarts, dim)
        )

        best_x, best_value = None, np.inf
        for start in starts:
            result = minimize(
                negative_expected_improvement, start, bounds=bounds, method="L-BFGS-B"
            )
            if result.fun < best_value:
                best_value = result.fun
                best_x = result.x

        x_next = np.clip(best_x, action_min_scaled, action_max_scaled)

        if movement_budget_remaining is not None:
            prev_x = X_train_scaled[-1].reshape(-1)
            scale = np.maximum(action_max_scaled - action_min_scaled, 1e-8)
            move_cost = np.linalg.norm((x_next - prev_x) / scale, ord=2)
            if move_cost > movement_budget_remaining and move_cost > 1e-12:
                ratio = max(movement_budget_remaining, 0.0) / move_cost
                x_next = prev_x + ratio * (x_next - prev_x)

        return x_next.reshape(1, -1)
