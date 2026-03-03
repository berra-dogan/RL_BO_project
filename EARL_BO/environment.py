import torch
import numpy as np
from turbo.turbo_1 import ModifiedTurbo1

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
        self.turbo = ModifiedTurbo1(X_train_scaled, scaler_EI.fit_transform(y_train_raw), self.lb, self.ub, device=self.device, verbose=True)


    def reset(self):
        # Reset the environment to its initial state
        self.X_train = self.X_train_scaled
        self.y_train = self.y_train_raw
        self.y_max = self.y_max_raw

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
        action = self.to_action(action_).reshape(1, -1)
        self.y_max = np.max(self.y_train)
        mean, std = self.model.predict(action, return_std=True)
        mean = self.scaler_EI.inverse_transform_mean(mean.reshape(-1, ))
        std = self.scaler_EI.inverse_transform_std(std.reshape(-1, ))

        # Simulate next observation
        next_observation = np.array([np.random.normal(mean[0], std[0])]).reshape(-1, )
        reward = max(0, (next_observation - self.y_max)[0])

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
