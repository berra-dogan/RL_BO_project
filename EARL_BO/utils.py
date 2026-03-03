import numpy as np

class Scaler:
    def __init__(self):
        self.mu = None
        self.std = None

    def fit(self, x):
        self.mu = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        # Avoid divide-by-zero and invalid inverse transforms.
        self.std = np.where(self.std == 0, 1.0, self.std)
        return self

    def fit_transform(self, x):
        self.fit(x)
        return (x - self.mu) / self.std

    def inverse_transform_mean(self, x):
        return x * self.std + self.mu

    def inverse_transform_std(self, x):
        return x * self.std

# Define the Memory class to store experience for the RL agent
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
