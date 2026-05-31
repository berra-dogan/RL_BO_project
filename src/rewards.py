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
    remaining_budget: float | None = None
    total_budget: float | None = None
    over_budget: float = 0.0

    @property
    def remaining_budget_fraction(self) -> float:
        if self.remaining_budget is None or self.total_budget is None or self.total_budget <= 0:
            return 1.0
        return float(np.clip(self.remaining_budget / self.total_budget, 0.0, 1.0))


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


def budgeted_exploration_reward(ctx: RewardContext, params: dict[str, float]) -> float:
    explore_weight = params.get("explore_weight", 1.0)
    path_cost_weight = params.get("path_cost_weight", 0.05)
    over_budget_penalty = params.get("over_budget_penalty", 5.0)

    remaining = max(ctx.remaining_budget_fraction, 0.0)

    exploration_bonus = remaining * explore_weight * max(ctx.std, 0.0)

    # Penalty grows sharply as budget gets low
    budget_pressure = 1.0 / max(remaining, 0.05)

    path_penalty = budget_pressure * path_cost_weight * ctx.move_cost

    budget_violation_penalty = over_budget_penalty * ctx.over_budget

    return (
        ctx.improvement
        + exploration_bonus
        - path_penalty
        - budget_violation_penalty
    )
REWARD_FUNCTIONS: dict[str, RewardFunction] = {
    "budgeted_exploration": budgeted_exploration_reward,
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
