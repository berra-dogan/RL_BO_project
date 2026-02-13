# config.py
from dataclasses import dataclass
from typing import List

# -------------------------------
# General settings
# -------------------------------
# @dataclass
# class GeneralConfig:
#     device: str = "cuda"  # or "cpu"
#     seed: int = 1
#     max_episodes: int = 4000
#     update_episode: int = 50
#     off_policy_episodes: int = 400
#     horizon: int = 5
#     no_improvement_threshold: int = 15  # episodes * update_episode
#     verbose: bool = True

# -------------------------------
# PPO Agent Hyperparameters
# -------------------------------
@dataclass
class PPOConfig:
    learning_rate: float = 0.001
    action_std: float = 0.1
    action_std_min: float = 0.01
    action_decay: float = 0.99
    K_epochs: int = 100
    eps_clip: float = 0.2
    gamma: float = 0.95
    gamma_increase: float = 1.0
    VF_coeff: float = 0.5
    Entropy_coeff: float = 0.01
    freeze_num: int = 2
    betas = (0.9, 0.999)

# -------------------------------
# Environment / BO settings
# -------------------------------
# @dataclass
# class EnvConfig:
#     num_state: int = 16  # dimension of encoded state
#     action_min: float = -1.0
#     action_max: float = 1.0
#     GP_lengthscale: float = 1.0
#     GP_noise_level: float = 1.0
#     GP_n_restarts_optimizer: int = 10

# -------------------------------
# RL Encoder settings
# -------------------------------
# @dataclass
# class EncoderConfig:
#     input_dim: int = 17  # X dimension + 1 for y
#     hidden_dim: int = 64
#     output_dim: int = 16
#     learning_rate: float = 0.01
#     betas: tuple = (0.9, 0.999)

# -------------------------------
# Test function / experiment settings
# -------------------------------
@dataclass
class ExperimentConfig:
    """Hyperparameters and experiment settings."""
    dimension: int = 30
    test_func_name: str = 'ackley'
    num_runs: int = 1
    num_experiments: int = 1
    num_initial_data: int = 30
    lower_bound: float = -2.0
    upper_bound: float = 2.0
    num_workers: int = 10
    horizon: int = 3
