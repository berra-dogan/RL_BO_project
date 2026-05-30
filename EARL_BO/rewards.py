from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class RewardContext:
    improvement: float
    move_cost: float
    mean: float
    std: float
    next_observation: float
    y_max: float
    action: np.ndarray
    prev_action: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray


RewardFunction = Callable[[RewardContext, dict[str, float]], float]


def earlbo_reward(ctx: RewardContext, params: dict[str, float]) -> float:
    return ctx.improvement


def snake_reward(ctx: RewardContext, params: dict[str, float]) -> float:
    path_cost_weight = params.get("path_cost_weight", 0.0)
    return ctx.improvement - path_cost_weight * ctx.move_cost


def log_improvement_reward(ctx: RewardContext, params: dict[str, float]) -> float:
    scale = params.get("scale", 1.0)
    return float(np.log1p(scale * ctx.improvement))


def normalized_improvement_reward(ctx: RewardContext, params: dict[str, float]) -> float:
    scale = max(float(np.max(ctx.upper_bound - ctx.lower_bound)), 1e-8)
    return ctx.improvement / scale


def optimistic_improvement_reward(ctx: RewardContext, params: dict[str, float]) -> float:
    std_weight = params.get("std_weight", 0.1)
    return max(0.0, ctx.mean + std_weight * ctx.std - ctx.y_max)


REWARD_FUNCTIONS: dict[str, RewardFunction] = {
    "earlbo": earlbo_reward,
    "snake": snake_reward,
    "log_improvement": log_improvement_reward,
    "normalized_improvement": normalized_improvement_reward,
    "optimistic_improvement": optimistic_improvement_reward,
}


def available_reward_modes() -> tuple[str, ...]:
    return tuple(sorted(REWARD_FUNCTIONS))


def get_reward_function(name: str) -> RewardFunction:
    try:
        return REWARD_FUNCTIONS[name]
    except KeyError as exc:
        choices = ", ".join(available_reward_modes())
        raise ValueError(f"Unknown reward mode '{name}'. Available modes: {choices}") from exc

