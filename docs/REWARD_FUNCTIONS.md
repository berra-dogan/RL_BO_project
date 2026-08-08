# Reward Functions

This document summarizes the reward functions implemented in
`src/rewards.py`. These rewards are used to train the EARLBO policy during
simulated lookahead. They do not directly change the black-box objective; they
change what the RL agent is encouraged to do while choosing the next BO point.

## Shared Terms

At each simulated lookahead step, the reward function receives a
`RewardContext`:

| Symbol | Code field | Meaning |
|---|---|---|
| `I` | `ctx.improvement` | Simulated improvement over the current best value: `max(0, next_observation - y_max)`. |
| `c` | `ctx.move_cost` | Movement cost from the previous point to the proposed point, measured in scaled search space. |
| `mu` | `ctx.mean` | GP predictive mean at the proposed point, transformed back to objective scale. |
| `sigma` | `ctx.std` | GP predictive standard deviation at the proposed point. |
| `y_best` | `ctx.y_max` | Best observed objective value so far. |
| `b_rem` | `ctx.remaining_budget` | Remaining movement budget, if a movement budget is active. |
| `b_total` | `ctx.total_budget` | Total movement budget, if a movement budget is active. |
| `rho` | `ctx.remaining_budget_fraction` | Remaining budget fraction, clipped to `[0, 1]`. |
| `c_over` | `ctx.over_budget` | Amount by which a proposed simulated move exceeds remaining movement budget. |
| `c_future` | `ctx.expected_future_move_cost` | Expected movement cost from the proposed point to likely future promising or uncertain points. |

The code treats the objective as a maximization problem. Lower regret means the
optimizer has found a value closer to the true optimum.

## `earlbo`

Formula:

```text
R = I
```

Code:

```python
return ctx.improvement
```

This is the base EARLBO reward. The agent is rewarded only when the simulated
next observation improves on the current best value.

Pros:

- Directly aligned with the BO objective of finding better function values.
- Simple and easy to interpret.
- Good baseline for comparing alternative reward shaping.

Cons:

- Sparse: many simulated points can produce zero improvement.
- Does not explicitly reward exploration in uncertain regions unless they
  happen to produce sampled improvement.
- Ignores movement cost, so it may jump far across the search space.

Best use:

- Baseline unconstrained Bayesian optimization.
- Cases where movement cost is irrelevant.

## `snake`

Formula:

```text
R = I - lambda_path * c
```

Code:

```python
path_cost_weight = params.get("path_cost_weight", 0.0)
return ctx.improvement - path_cost_weight * ctx.move_cost
```

This extends base EARLBO by penalizing movement. It encourages the agent to
prefer improvements that are close to the current point.

Pros:

- Adds a simple movement-cost tradeoff.
- Useful when far jumps are undesirable.
- Easy to tune through `snake_path_cost_weight`.

Cons:

- Penalizes each move independently, but does not enforce a total movement
  budget.
- Can become too conservative if `lambda_path` is too large.
- Does not distinguish between early and late budget usage.

Best use:

- Smoothly discouraging long movements.
- A simple movement-aware baseline before using a hard budget.

## `log_improvement`

Formula:

```text
R = log(1 + alpha * I)
```

Code:

```python
scale = params.get("scale", 1.0)
return log1p(scale * ctx.improvement)
```

This rewards improvement, but compresses large improvements using a logarithm.
Small improvements still matter, while very large simulated improvements have
reduced influence.

Pros:

- Can stabilize PPO training by reducing reward spikes.
- Preserves the ordering of positive improvements.
- Less sensitive to rare large simulated observations.

Cons:

- Does not fundamentally change the optimization objective.
- The scale parameter `alpha` affects reward magnitude and training dynamics.
- Still gives no reward for uncertainty unless it leads to sampled improvement.

Best use:

- When raw improvement rewards are noisy or have occasional large values.
- As a reward-scaling alternative to base EARLBO.

## `normalized_improvement`

Formula:

```text
R = I / s
```

Current implementation:

```text
s = max(max(upper_bound - lower_bound), 1e-8)
```

Code:

```python
scale = max(float(np.max(ctx.upper_bound - ctx.lower_bound)), 1e-8)
return ctx.improvement / scale
```

This rescales improvement by the largest single-dimension search-space width.
With bounds `[-1, 1]`, the scale is `2`, so the reward is currently
`improvement / 2`.

Pros:

- Keeps reward magnitudes smaller than raw improvement.
- Can improve PPO stability through gentler reward scaling.
- Simple to compare with base EARLBO.

Cons:

- In the current implementation, it does not meaningfully account for
  dimension. For `[-1, 1]^10` and `[-1, 1]^30`, the scale is still `2`.
- It is mostly a constant rescaling of EARLBO when bounds are fixed.
- Any improvement over `earlbo` likely comes from PPO training dynamics, not a
  new acquisition objective.

Best use:

- Reward-scale sensitivity tests.
- As a simple check for whether PPO benefits from smaller reward magnitudes.

Possible improvement:

```text
s = ||upper_bound - lower_bound||_2
```

For `[-1, 1]^d`, this would give:

```text
s = 2 * sqrt(d)
```

which would make the normalization dimension-aware.

## `optimistic_improvement`

Formula:

```text
R = max(0, mu + beta * sigma - y_best)
```

Code:

```python
std_weight = params.get("std_weight", 0.1)
return max(0.0, ctx.mean + std_weight * ctx.std - ctx.y_max)
```

This is an optimistic reward similar in spirit to an upper-confidence-bound
criterion. It rewards points that either have high predicted mean or high
uncertainty.

Pros:

- Explicitly encourages exploration through predictive uncertainty.
- Less dependent on a lucky simulated observation than raw improvement.
- Can help the agent explore regions that are uncertain but not yet known to be
  good.

Cons:

- The `std_weight` parameter controls the exploration/exploitation balance and
  must be tuned.
- Too much uncertainty weight can over-explore.
- It optimizes a GP optimism proxy, not direct sampled improvement.

Best use:

- When base EARLBO is too exploitative or gets stuck.
- When uncertainty-driven exploration is important.

## `budgeted_exploration`

Formula:

```text
R = I
    + rho * w_explore * max(sigma, 0)
    - (1 / max(rho, 0.05)) * w_path * c
    - w_over * c_over
```

where:

```text
rho = remaining_budget / total_budget
```

Code:

```python
exploration_bonus = remaining * explore_weight * max(ctx.std, 0.0)
budget_pressure = 1.0 / max(remaining, 0.05)
path_penalty = budget_pressure * path_cost_weight * ctx.move_cost
budget_violation_penalty = over_budget_penalty * ctx.over_budget

return (
    ctx.improvement
    + exploration_bonus
    - path_penalty
    - budget_violation_penalty
)
```

This reward is designed for BO with a movement budget. It encourages
uncertainty-driven exploration while budget remains, penalizes movement more
strongly as budget gets low, and penalizes moves that exceed the remaining
budget during simulated lookahead.

In addition to the reward, `src/rl_bo.py` projects the final selected point back
into the remaining movement budget:

```python
x_next = env.project_to_remaining_budget(x_next)
```

So the movement budget is enforced at action selection time.

Pros:

- Adds explicit budget awareness.
- Balances improvement, uncertainty, and movement cost.
- Can enforce a hard movement limit through final-action projection.
- Useful for comparing BO methods under both evaluation and movement budgets.

Cons:

- More parameters to tune:
  - `movement_budget`
  - `reward_param_explore_weight`
  - `reward_param_path_cost_weight`
  - `reward_param_over_budget_penalty`
- The budget is treated as a maximum, not a target. The method may leave budget
  unused if additional movement does not appear worthwhile.
- If path cost is weighted too strongly, it can become too conservative.
- If exploration weight is too high, it may spend budget on uncertainty without
  enough improvement.

Best use:

- Constrained BO where moving between points has a cost.
- Comparing regret under a fixed movement budget.

Ways to encourage fuller budget use:

- Penalize unused budget in the tuning score rather than penalizing total
  movement cost.
- Add a small movement-use bonus while budget remains.
- Reduce `reward_param_path_cost_weight`.
- Add a terminal penalty for unused budget near the end of the BO horizon.

## `lookahead_budgeted_exploration`

Purpose:

`budgeted_exploration` is myopic: it asks whether the next point is useful and
cheap to reach from the current point. `lookahead_budgeted_exploration` adds one
extra question: after moving to this proposed point, would the agent be well
positioned for plausible future evaluations?

The reward does not run a full multi-step Bayesian optimization rollout.
Instead, it computes a lightweight proxy for future travel cost under the
current GP posterior. This makes the reward path-aware without making training
as expensive as nested lookahead acquisition optimization.

Formula:

```text
R = I
    + rho * w_explore * max(sigma, 0)
    - (1 / max(rho, 0.05)) * w_path * c
    - (1 / max(rho, 0.05)) * w_future * c_future
    - w_over * c_over
```

where:

```text
rho = remaining_budget / total_budget
budget_pressure = 1 / max(rho, 0.05)
```

Expanded:

```text
R = immediate_improvement
    + current_uncertainty_bonus
    - immediate_movement_penalty
    - expected_future_movement_penalty
    - over_budget_penalty
```

with:

```text
immediate_improvement = I
current_uncertainty_bonus = rho * w_explore * max(sigma(x_next), 0)
immediate_movement_penalty = budget_pressure * w_path * d(x_prev, x_next)
expected_future_movement_penalty = budget_pressure * w_future * c_future
over_budget_penalty = w_over * c_over
```

The key addition is `c_future`. It estimates how far the proposed next point
`x_next` is from likely future regions of interest:

```text
c_future = sum_i p_i * d(x_next, z_i)
```

where:

- `z_i` is a randomly sampled candidate future point from the search space.
- `d(x_next, z_i)` is scaled Euclidean distance.
- `p_i` is the probability weight assigned to future candidate `z_i`.

How future candidates are generated:

```python
candidates = np.random.uniform(lower_bound, upper_bound, size=(n_candidates, d))
```

The implementation samples candidate future points uniformly from the whole
search space. It then asks the current GP model how promising each candidate
looks. A candidate is promising if it has a high posterior mean, high posterior
uncertainty, or both.

Future candidate score:

```text
s_i = max(0, mu(z_i) + beta_future * sigma(z_i) - y_best)
```

This is an optimistic-improvement score. It is zero for candidates whose
optimistic GP value is still below the current best observation.

Future candidate weights:

```text
p_i = exp(s_i / tau) / sum_j exp(s_j / tau)
```

If all future scores are essentially zero, the implementation falls back to
uniform weights:

```text
p_i = 1 / n_candidates
```

So the full future movement penalty is:

```text
expected_future_path_penalty
    = budget_pressure
      * w_future
      * sum_i [
          exp(s_i / tau) / sum_j exp(s_j / tau)
        ] * d(x_next, z_i)
```

This reward keeps the original budgeted exploration terms, but adds a penalty
for choosing a point that would leave the agent far away from likely future
regions of interest. A future region is considered interesting if it has a high
GP mean, high GP uncertainty, or both. The reward therefore encourages the agent
to choose points that are useful immediately and also position the exploration
path well for plausible future evaluations.

Interpretation:

- If a proposed point has high immediate improvement but leaves the agent far
  away from other promising regions, the future penalty reduces its reward.
- If a proposed point is moderately good and also sits near several plausible
  future candidates, it can receive a better reward because it preserves future
  mobility.
- As the movement budget decreases, `budget_pressure` increases, so both
  immediate and expected future movement become more expensive.
- Early in the run, when `rho` is high, the reward allows more movement because
  the agent still has budget left.
- Late in the run, when `rho` is low, the reward becomes more conservative.

Parameters:

- `reward_param_explore_weight`: controls the uncertainty bonus at the selected
  point. Larger values make the agent more willing to sample uncertain regions.
- `reward_param_path_cost_weight`: controls the immediate movement penalty from
  the previous point to the proposed point.
- `reward_param_future_path_cost_weight`: controls the expected future movement
  penalty. Larger values make the agent prefer points that keep it close to
  likely future regions of interest.
- `reward_param_future_num_candidates`: number of sampled possible future points
  used to estimate the future movement cost. The current tuning grid does not
  vary this parameter unless it is explicitly added; the implementation default
  is `128`.
- `reward_param_future_optimism_weight`: the value of `beta_future`, controlling
  how strongly GP uncertainty affects future-point attractiveness. Larger values
  make uncertain future candidates look more attractive.
- `reward_param_future_softmax_temperature`: the value of `tau`; lower values
  focus the expectation on the most attractive future points, while higher
  values spread weight more evenly across candidates. Very low values make
  `c_future` behave closer to distance from the single most promising sampled
  future point.
- `reward_param_over_budget_penalty`: controls the penalty for simulated moves
  that exceed the remaining movement budget.

Current fine-tuning grid:

In `src/experiments/reward_configs.py`, the current grid is:

```text
movement_budget: [5]
reward_param_explore_weight: [3.0, 5.0]
reward_param_path_cost_weight: [0.1]
reward_param_future_path_cost_weight: [0.005, 0.01]
reward_param_future_optimism_weight: [1.0, 0.5]
reward_param_future_softmax_temperature: [1.0, 0.5, 2.0]
reward_param_over_budget_penalty: [0.0]
```

That gives:

```text
2 * 1 * 2 * 2 * 3 * 1 = 24 reward-parameter configurations
```

For leave-one-function-out tuning with one held-out objective, each config is
tested on four training functions, so one held-out fold has:

```text
24 configs * 4 training functions = 96 tuning jobs
```

Best use:

- BO problems where the order of evaluations matters because movement resources
  are limited.
- Settings where moving to a distant point can be reasonable only if it also
  leaves the agent well positioned for likely future evaluations.
- Comparing myopic budget-aware exploration against a path-aware reward that
  estimates downstream travel cost.

Limitations:

- See `docs/LIMITATIONS.md` for the main limitation of this reward: the expected
  future path penalty is computed under the current GP posterior and does not
  simulate the posterior update that would occur after evaluating the proposed
  next point.
- The future candidates are randomly sampled, so the future-cost estimate has
  Monte Carlo noise.
- The estimate depends on the current GP model. If the GP posterior is poor,
  the reward can prefer misleading future regions.
- The reward estimates future travel cost, not future objective improvement
  after retraining the GP.

## Summary Table

| Reward | Formula | Main idea | Main risk |
|---|---|---|---|
| `earlbo` | `I` | Reward direct improvement. | Sparse and movement-unaware. |
| `snake` | `I - lambda_path c` | Reward improvement but penalize movement. | Can become too conservative. |
| `log_improvement` | `log(1 + alpha I)` | Compress improvement rewards. | Mostly reward scaling, not a new objective. |
| `normalized_improvement` | `I / s` | Normalize reward magnitude. | Current scale is not dimension-aware. |
| `optimistic_improvement` | `max(0, mu + beta sigma - y_best)` | Reward optimistic GP potential. | Can over-explore if `beta` is too high. |
| `budgeted_exploration` | `I + rho w_e sigma - pressure w_p c - w_o c_over` | Use improvement and uncertainty under movement budget. | More parameters; may under-use budget. |
| `lookahead_budgeted_exploration` | `I + rho w_e sigma - pressure w_p c - pressure w_f c_future - w_o c_over` | Penalize immediate movement and expected future movement. | Extra sampling and more parameters to tune. |

## Interpretation Guidance

When comparing these rewards, use both optimization quality and movement cost:

```text
final regret
average regret
standard deviation across runs
total scaled movement cost
regret at fixed movement budget
```

For reward functions that only rescale improvement, such as
`normalized_improvement`, improvements in results may come from PPO training
stability rather than from a different optimization objective. Use repeated test
runs before claiming a significant improvement.
