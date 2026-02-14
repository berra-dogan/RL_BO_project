import numpy as np

class Scaler:
    def __init__(self):
        self.mu = None
        self.std = None

    def fit_transform(self, x):
        self.mu = np.mean(x, 0)
        self.std = np.std(x, 0)

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
