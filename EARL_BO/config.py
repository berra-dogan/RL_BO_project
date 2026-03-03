# config.py
from dataclasses import dataclass, field

import torch

# -------------------------------
# General settings
# -------------------------------
@dataclass
class RLBOConfig:
    max_episodes: int = 100
    update_episode: int = 10
    off_policy_episodes: int = 40
    no_improvement_threshold: int = 15

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
# Experiment settings
# -------------------------------
@dataclass
class GPRConfig:
    rbf_length_scale: float = 1.0
    rbf_length_scale_bounds: tuple[float, float] = (1e-6, 1e2)
    wk_noise_level: float = 1.0
    wk_noise_level_bounds: tuple[float, float] = (1e-12, 1e1)
    alpha: float = 1e-10
    n_restarts_optimizer: int = 10

@dataclass
class ExperimentConfig:
    """Hyperparameters and experiment settings."""
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dimension: int = 30
    test_func_name: str = 'ackley'
    num_runs: int = 1
    num_experiments: int = 1
    num_initial_data: int = 30
    lower_bound: float = -2.0
    upper_bound: float = 2.0
    num_workers: int = 10
    horizon: int = 3
    gpr_config: GPRConfig = field(default_factory=GPRConfig) # A new config is created each time a new Experiment config is created
