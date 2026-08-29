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
    expected_future_move_cost: float = 0.0

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


def log_improvement_movement_cost_reward(ctx: RewardContext, params: dict[str, float]) -> float:
    scale = params.get("scale", 1.0)
    path_cost_weight = params.get("path_cost_weight", 0.0)
    return float(np.log1p(scale * ctx.improvement)) - path_cost_weight * ctx.move_cost


def normalized_improvement_reward(ctx: RewardContext, params: dict[str, float]) -> float:
    scale = max(float(np.max(ctx.upper_bound - ctx.lower_bound)), 1e-8)
    return ctx.improvement / scale


def optimistic_improvement_reward(ctx: RewardContext, params: dict[str, float]) -> float:
    std_weight = params.get("std_weight", 0.1)
    return max(0.0, ctx.mean + std_weight * ctx.std - ctx.y_max)


def optimistic_improvement_movement_cost_reward(
    ctx: RewardContext, params: dict[str, float]
) -> float:
    std_weight = params.get("std_weight", 0.1)
    path_cost_weight = params.get("path_cost_weight", 0.0)
    optimistic_improvement = max(0.0, ctx.mean + std_weight * ctx.std - ctx.y_max)
    return optimistic_improvement - path_cost_weight * ctx.move_cost


def _normalized_move_cost(ctx: RewardContext) -> float:
    """ctx.move_cost is an L2 distance in per-dimension range-normalized action
    units, so it lives in [0, sqrt(dim)]. Divide by sqrt(dim) to get a value in
    [0, 1] that is comparable across dimensions and to an O(1) reward term."""
    dim = max(int(np.asarray(ctx.action).size), 1)
    return float(ctx.move_cost) / float(np.sqrt(dim))


def optimistic_improvement_movement_cost2_reward(
    ctx: RewardContext, params: dict[str, float]
) -> float:
    """Like optimistic_improvement_movement_cost, but the penalty is (a)
    normalized to [0, 1] via _normalized_move_cost and (b) scaled by the size of
    the improvement term so it stays a *relative* trade-off instead of an
    absolute subtraction that vanishes once the improvement signal is small."""
    std_weight = params.get("std_weight", 0.1)
    path_cost_weight = params.get("path_cost_weight", 0.0)
    optimistic_improvement = max(0.0, ctx.mean + std_weight * ctx.std - ctx.y_max)
    penalty = (
        path_cost_weight
        * _normalized_move_cost(ctx)
        * max(optimistic_improvement, 1.0)
    )
    return optimistic_improvement - penalty


def log_improvement_movement_cost2_reward(
    ctx: RewardContext, params: dict[str, float]
) -> float:
    """Like log_improvement_movement_cost, but with the normalized, relative
    movement penalty from optimistic_improvement_movement_cost2_reward."""
    scale = params.get("scale", 1.0)
    path_cost_weight = params.get("path_cost_weight", 0.0)
    base = float(np.log1p(scale * ctx.improvement))
    penalty = path_cost_weight * _normalized_move_cost(ctx) * max(base, 1.0)
    return base - penalty


def optimistic_improvement_movement_cost3_reward(
    ctx: RewardContext, params: dict[str, float]
) -> float:
    """Copy of optimistic_improvement_movement_cost2_reward, kept as a separate
    reward mode so its penalty shaping can be varied independently."""
    std_weight = params.get("std_weight", 0.1)
    path_cost_weight = params.get("path_cost_weight", 0.0)
    optimistic_improvement = max(0.0, ctx.mean + std_weight * ctx.std - ctx.y_max)
    penalty = (
        path_cost_weight
        * _normalized_move_cost(ctx)
        * max(optimistic_improvement, 1.0)
    )
    return optimistic_improvement - penalty


def log_improvement_movement_cost3_reward(
    ctx: RewardContext, params: dict[str, float]
) -> float:
    """Copy of log_improvement_movement_cost2_reward, kept as a separate reward
    mode so its penalty shaping can be varied independently."""
    scale = params.get("scale", 1.0)
    path_cost_weight = params.get("path_cost_weight", 0.0)
    base = float(np.log1p(scale * ctx.improvement))
    penalty = path_cost_weight * _normalized_move_cost(ctx) * max(base, 1.0)
    return base - penalty


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


def lookahead_budgeted_exploration_reward(ctx: RewardContext, params: dict[str, float]) -> float:
    explore_weight = params.get("explore_weight", 1.0)
    path_cost_weight = params.get("path_cost_weight", 0.05)
    future_path_cost_weight = params.get("future_path_cost_weight", 0.05)
    over_budget_penalty = params.get("over_budget_penalty", 5.0)

    remaining = max(ctx.remaining_budget_fraction, 0.0)
    exploration_bonus = remaining * explore_weight * max(ctx.std, 0.0)
    budget_pressure = 1.0 / max(remaining, 0.05)

    path_penalty = budget_pressure * path_cost_weight * ctx.move_cost
    future_path_penalty = (
        budget_pressure
        * future_path_cost_weight
        * ctx.expected_future_move_cost
    )
    budget_violation_penalty = over_budget_penalty * ctx.over_budget

    return (
        ctx.improvement
        + exploration_bonus
        - path_penalty
        - future_path_penalty
        - budget_violation_penalty
    )


REWARD_FUNCTIONS: dict[str, RewardFunction] = {
    "budgeted_exploration": budgeted_exploration_reward,
    "earlbo": earlbo_reward,
    "lookahead_budgeted_exploration": lookahead_budgeted_exploration_reward,
    "snake": snake_reward,
    "log_improvement": log_improvement_reward,
    "log_improvement_movement_cost": log_improvement_movement_cost_reward,
    "log_improvement_movement_cost2": log_improvement_movement_cost2_reward,
    "log_improvement_movement_cost3": log_improvement_movement_cost3_reward,
    "normalized_improvement": normalized_improvement_reward,
    "optimistic_improvement": optimistic_improvement_reward,
    "optimistic_improvement_movement_cost": optimistic_improvement_movement_cost_reward,
    "optimistic_improvement_movement_cost2": optimistic_improvement_movement_cost2_reward,
    "optimistic_improvement_movement_cost3": optimistic_improvement_movement_cost3_reward,
}


def available_reward_modes() -> tuple[str, ...]:
    return tuple(sorted(REWARD_FUNCTIONS))


def get_reward_function(name: str) -> RewardFunction:
    try:
        return REWARD_FUNCTIONS[name]
    except KeyError as exc:
        choices = ", ".join(available_reward_modes())
        raise ValueError(f"Unknown reward mode '{name}'. Available modes: {choices}") from exc
