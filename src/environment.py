import torch
import numpy as np
from turbo.turbo_1 import ModifiedTurbo1
from rewards import RewardContext, get_reward_function

class Env_encoder:
    def __init__(
        self,
        model,
        encoder,
        y_max_raw,
        X_train_scaled,
        y_train_raw,
        scaler_EI,
        action_min_scaled,
        action_max_scaled,
        reward_mode,
        snake_path_cost_weight,
        reward_params,
        movement_budget_remaining,
        movement_budget_total,
        device,
    ):
        # Initialize the environment in scaled feature space.
        self.device = device
        
        self.num_state = 16
        self.num_action = X_train_scaled.shape[1]
        self.action_min = action_min_scaled
        self.action_max = action_max_scaled
        self.lb = np.min(X_train_scaled, axis=0)
        self.ub = np.max(X_train_scaled, axis=0)
        self.encoder = encoder
        self.model = model
        self.X_train_scaled = X_train_scaled
        self.y_train_raw = y_train_raw
        self.scaler_EI = scaler_EI
        self.y_max_raw = y_max_raw
        self.reward_mode = reward_mode
        self.reward_params = dict(reward_params or {})
        self.reward_params.setdefault("path_cost_weight", snake_path_cost_weight)
        self.reward_function = get_reward_function(reward_mode)
        self.initial_budget_remaining = movement_budget_remaining
        self.movement_budget_remaining = movement_budget_remaining
        self.movement_budget_total = movement_budget_total
        self.turbo = ModifiedTurbo1(X_train_scaled, scaler_EI.fit_transform(y_train_raw), self.lb, self.ub, device=self.device, verbose=True)


    def reset(self):
        # Reset the environment to its initial state
        self.X_train = self.X_train_scaled
        self.y_train = self.y_train_raw
        self.y_max = self.y_max_raw
        self.movement_budget_remaining = self.initial_budget_remaining

        # Fit the scaler and model
        y_train_scaled = self.scaler_EI.fit_transform(self.y_train)
        self.model.fit(self.X_train, y_train_scaled)

        # Encode the initial state
        X_train_org_pt = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y_train_org_pt = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
        encoder_input_pt = torch.cat([X_train_org_pt, y_train_org_pt], dim=1)
        self.state = self.encoder(encoder_input_pt)

        return self.state.to(self.device)

    def step(self, action_):
        # Perform one step in the environment
        prev_action = self.X_train[-1].reshape(-1)
        action = self.to_action(action_).reshape(1, -1)
        self.y_max = np.max(self.y_train)
        mean, std = self.model.predict(action, return_std=True)
        mean = self.scaler_EI.inverse_transform_mean(mean.reshape(-1, ))
        std = self.scaler_EI.inverse_transform_std(std.reshape(-1, ))

        # Simulate next observation
        next_observation = np.array([np.random.normal(mean[0], std[0])]).reshape(-1, )
        improvement = max(0, (next_observation - self.y_max)[0])
        next_action = action.reshape(-1)
        scale = self.action_scale()
        move_cost = np.linalg.norm((next_action - prev_action) / scale, ord=2)
        expected_future_move_cost = self.estimate_expected_future_move_cost(next_action, scale)
        over_budget = 0.0
        if self.movement_budget_remaining is not None:
            over_budget = max(0.0, move_cost - self.movement_budget_remaining)
        reward_context = RewardContext(
            improvement=float(improvement),
            move_cost=float(move_cost),
            mean=float(mean[0]),
            std=float(std[0]),
            next_observation=float(next_observation[0]),
            y_max=float(self.y_max),
            action=next_action,
            prev_action=prev_action,
            lower_bound=self.lb,
            upper_bound=self.ub,
            remaining_budget=self.movement_budget_remaining,
            total_budget=self.movement_budget_total,
            over_budget=float(over_budget),
            expected_future_move_cost=float(expected_future_move_cost),
        )
        reward = float(self.reward_function(reward_context, self.reward_params))
        if self.movement_budget_remaining is not None:
            self.movement_budget_remaining = max(0.0, self.movement_budget_remaining - move_cost)

        # Update training data
        self.X_train = np.vstack((self.X_train, action))
        self.y_train = np.vstack((self.y_train, next_observation))

        # ADD THIS SECTION: Update GP model with new observation
        y_train_scaled = self.scaler_EI.fit_transform(self.y_train)
        self.model.fit(self.X_train, y_train_scaled)

        # Encode the next state
        X_train_org_pt = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y_train_org_pt = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
        encoder_input_pt = torch.cat([X_train_org_pt, y_train_org_pt], dim=1)
        next_state = self.encoder(encoder_input_pt)

        self.state = next_state

        return next_state.to(self.device), reward

    def to_action(self, action):
        # Convert normalized action to actual action space
        act_k = (self.action_max - self.action_min) / 2.
        act_b = (self.action_max + self.action_min) / 2.
        return np.array(act_k * action + act_b).reshape(1, -1)

    def reverse_action(self, action):
        # Convert actual action to normalized action space
        act_k_inv = 2. / (self.action_max - self.action_min)
        act_b = (self.action_max + self.action_min) / 2.
        return np.array(act_k_inv * (action - act_b)).reshape(1, -1)

    def turbo_acquisition(self):
        # Use TuRBO for acquisition
        action = self.turbo.optimize()
        return np.array(action).flatten()

    def action_scale(self):
        return np.maximum(self.action_max - self.action_min, 1e-8)

    def estimate_expected_future_move_cost(self, action, scale):
        default_future_weight = (
            0.05 if self.reward_mode == "lookahead_budgeted_exploration" else 0.0
        )
        future_weight = self.reward_params.get("future_path_cost_weight", default_future_weight)
        if future_weight <= 0.0:
            return 0.0

        n_candidates = int(self.reward_params.get("future_num_candidates", 128))
        if n_candidates <= 0:
            return 0.0

        optimism = self.reward_params.get("future_optimism_weight", 1.0)
        temperature = max(self.reward_params.get("future_softmax_temperature", 1.0), 1e-8)

        candidates = np.random.uniform(self.lb, self.ub, size=(n_candidates, self.num_action))
        means, stds = self.model.predict(candidates, return_std=True)
        means = self.scaler_EI.inverse_transform_mean(means.reshape(-1,))
        stds = self.scaler_EI.inverse_transform_std(stds.reshape(-1,))

        future_scores = np.maximum(0.0, means + optimism * stds - self.y_max)
        if np.all(future_scores <= 1e-12):
            weights = np.full(n_candidates, 1.0 / n_candidates)
        else:
            logits = future_scores / temperature
            logits = logits - np.max(logits)
            weights = np.exp(logits)
            weights = weights / np.sum(weights)

        distances = np.linalg.norm((candidates - action.reshape(1, -1)) / scale, ord=2, axis=1)
        return float(np.dot(weights, distances))

    def move_cost(self, action):
        prev_action = self.X_train_scaled[-1].reshape(-1)
        next_action = np.array(action).reshape(-1)
        return float(np.linalg.norm((next_action - prev_action) / self.action_scale(), ord=2))

    def project_to_remaining_budget(self, action):
        if self.initial_budget_remaining is None:
            return np.array(action).reshape(1, -1)

        action = np.array(action).reshape(-1)
        prev_action = self.X_train_scaled[-1].reshape(-1)
        move_cost = self.move_cost(action)
        if move_cost <= self.initial_budget_remaining or move_cost <= 1e-12:
            return action.reshape(1, -1)

        ratio = max(self.initial_budget_remaining, 0.0) / move_cost
        return (prev_action + ratio * (action - prev_action)).reshape(1, -1)
