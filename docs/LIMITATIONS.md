# Limitations

## Lookahead Budgeted Exploration

The `lookahead_budgeted_exploration` reward estimates whether a proposed next
point leaves the optimizer well positioned for likely future evaluations. It
does this by scoring possible future points under the current GP posterior and
penalizing the expected movement cost from the proposed next point to those
future points.

This gives the reward a path-aware component, but it is not a full Bayesian
lookahead.

### No Posterior Update For The Candidate Point

The current implementation estimates future regions using the GP posterior
before the proposed point is evaluated.

It asks:

```text
Given what is known now, where might the optimizer want to go later?
```

It does not ask:

```text
If this proposed point were evaluated first, how would the GP posterior change,
and where would the optimizer want to go after that?
```

This means the reward does not account for the uncertainty reduction caused by
evaluating the proposed point.

For example, suppose there are two regions:

```text
A = nearby, moderately useful or uncertain region
B = farther, better-looking but also uncertain region
```

The reward can encourage visiting `A` before `B` if `A` looks like a likely
future destination and visiting `B` first would require backtracking. This is the
intended path-aware behavior.

However, if evaluating `B` would reduce uncertainty in the surrounding region,
the current reward does not model that posterior update. It may still treat
nearby points around `B` as future-attractive because their uncertainty is high
under the current GP.

So the current future-cost estimate is:

```text
score future points using the GP before observing x_next
```

not:

```text
fantasize observing x_next,
update the GP posterior,
then score future points using the updated GP
```

### Practical Consequence

The reward can reduce unnecessary backtracking when the current GP already
identifies likely future regions. It is less accurate when the value of a move
comes mainly from how much that move would reduce local uncertainty after the
observation is made.

In particular, it may overestimate the need to keep exploring around a proposed
point because it does not know that evaluating that point would make nearby
points less uncertain.

### More Complete Alternative

A more complete version would use a fantasy posterior:

1. Consider a proposed next point `x_next`.
2. Create one or more fantasy observations at `x_next`, for example by using the
   GP predicted mean or samples from the predictive distribution.
3. Temporarily update a copy of the GP with `(x_next, fantasy_y)`.
4. Recompute future-point scores under the updated GP.
5. Compute the expected future movement cost using that updated future
   distribution.

This would better capture the fact that evaluating a point reduces uncertainty
near that point.

The downside is computational cost. The current implementation estimates future
movement cost with one GP prediction over sampled future points. A fantasy
posterior version would require additional GP updates or posterior computations
during reward evaluation, which would make training substantially slower.

### Interpretation

Results from `lookahead_budgeted_exploration` should therefore be interpreted as
testing a lightweight path-aware reward, not a full multi-step Bayesian
lookahead acquisition function.

It models:

```text
future movement cost under the current belief
```

but not:

```text
future movement cost after accounting for the information gained by the next
evaluation
```

```text
IDEA: Instead of sampling next points uniformly between lower and upper bounds we could sample them in a way that the close points to the candidate next point will be less likely as the std in that area is expected to decrease as we sample the next point.
```
```text
IDEA: We could also add a penalty to the sampled future point scores depending on their closeness to the next point to be sampled. 
```
```text
EXPLORE: Tmax
```