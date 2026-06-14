# Fine Tune Each Reward Function With PBS Array Jobs

This workflow uses the PBS array-job pattern from `HPC Array Jobs_ An Intro.pdf`.
Each array element fine-tunes one reward function by running `src/experiments/run_one_reward_experiments.py`
with a separate output directory.

## 1. Confirm Reward Functions

Reward functions are registered in `src/rewards.py`.

Current reward modes:

```text
earlbo
snake
log_improvement
normalized_improvement
optimistic_improvement
budgeted_exploration
lookahead_budgeted_exploration
```

`budgeted_exploration` is intentionally tuned separately with
`src/experiments/run_budgeted_exploration_tuning.py`.
The `lookahead_budgeted_exploration` reward is a separate path-aware reward
defined in `docs/REWARD_FUNCTIONS.md` and is intentionally tuned separately
with `src/experiments/run_lookahead_budgeted_exploration_tuning.py`.

To add another reward:

1. Add a function in `src/rewards.py`.
2. Add it to `REWARD_FUNCTIONS`.
3. The new name will automatically appear in `--reward-mode`.

## 2. Choose The Hyperparameter Search Space

Edit `SEARCH_SPACE` in `src/experiments/run_one_reward_experiments.py`.

Example:

```python
SEARCH_SPACE = {
    "max_episodes": [300],
    "off_policy_episodes": [20, 40],
    "encoder_learning_rate": [1e-3],
    "ppo_learning_rate": [1e-4],
    "ppo_action_std": [0.1],
    "ppo_action_decay": [0.99],
    "ppo_gamma": [0.95],
    "ppo_entropy_coeff": [0.01],
    "gpr_alpha": [1e-8],
}
```

Do not put `reward_mode` in `SEARCH_SPACE` for this PBS workflow. The array
index selects the reward mode.

Set the tuning budget in the same file:

```python
TUNE_BUDGET = {"num_runs": 3, "num_experiments": 15}
```

Use a small budget first, then increase it once the PBS job works.

## 3. Sync Code To The Cluster

Run this on your Mac:

```bash
cd /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation
./scripts/sync_to_cluster.sh
```

Then SSH:

```bash
ssh bd225@login.cx3.hpc.imperial.ac.uk
```

## 4. PBS Scripts

PBS scripts are checked in under `cluster/`; do not create ad hoc copies in the
repository root.

```text
cluster/submit_reward_config_array.pbs
cluster/group_submit_reward_finetune.pbs
cluster/submit_reward_tests.pbs
cluster/submit_budgeted_exploration_param_array.pbs
cluster/submit_budgeted_exploration_param_tuning.pbs
```

Use `submit_reward_config_array.pbs` for the full reward/config sweep, and use
`submit_budgeted_exploration_param_tuning.pbs` for the compact grouped
budgeted-exploration sweep.

## 5. Submit A Small Test First

From the cluster login node:

```bash
cd ~/my_implementation
qsub -J 0-1 cluster/submit_reward_config_array.pbs
```

Monitor it:

```bash
qstat -u $USER -t
```

After it finishes, check output:

```bash
find output/reward_finetune_reward_params -name best_config.json
```

If the test works, submit all rewards:

```bash
qsub cluster/submit_reward_config_array.pbs
```

## 6. Monitor And Debug

List all array jobs:

```bash
qstat -u $USER -t
```

Inspect logs for a specific array element:

```bash
cat reward_finetune.o<JOB_ID>[0]
cat reward_finetune.e<JOB_ID>[0]
```

Cancel the full job:

```bash
qdel <JOB_ID>
```

Cancel one array element:

```bash
qdel <JOB_ID>[3]
```

## 7. Resume Safely After Failures

The command uses `--skip-existing`, so completed tuning runs are skipped when
you resubmit.

Resubmit the same PBS script after a node failure or walltime kill:

```bash
qsub cluster/submit_reward_config_array.pbs
```

Check completed reward searches:

```bash
find output/reward_finetune_reward_params -path "*/tuning/best_config.json" -print
```

Expected outputs:

```text
output/reward_finetune_reward_params/earlbo/tuning/best_config.json
output/reward_finetune_reward_params/snake/tuning/best_config.json
output/reward_finetune_reward_params/log_improvement/tuning/best_config.json
output/reward_finetune_reward_params/normalized_improvement/tuning/best_config.json
output/reward_finetune_reward_params/optimistic_improvement/tuning/best_config.json
```

## 8. Compare The Best Reward Configs

Each reward writes:

```text
output/reward_finetune_reward_params/<reward>/tuning/best_config.json
output/reward_finetune_reward_params/<reward>/tuning/tuning_results.csv
output/reward_finetune_reward_params/<reward>/tuning/best_tuning/
```

The most important fields in `best_config.json` are:

```text
score
final_regret
best_regret
params
base_settings
saved_summary_csv
```

Lower regret is better.

## 9. Run Final Tests For Each Tuned Reward

After tuning, run the final test phase for each reward. You can use another
array job with the same reward list, but change `--mode tune` to `--mode test`:

```bash
python src/experiments/run_one_reward_experiments.py \
  --mode test \
  --device cpu \
  --output-root "reward_finetune/${REWARD}" \
  --reward-mode "$REWARD" \
  $EXTRA_ARGS \
  --skip-existing
```

This loads:

```text
output/reward_finetune_reward_params/<reward>/tuning/best_config.json
```

and writes the final result under:

```text
output/reward_finetune_reward_params/<reward>/test_best/
```

## 10. Copy Results Back To Your Mac

Run this on your Mac:

```bash
cd /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation
rsync -av bd225@login.cx3.hpc.imperial.ac.uk:~/my_implementation/output/reward_finetune_reward_params/ output/reward_finetune_reward_params/
```

## 11. Calculate Movement Cost Used By Best Configs

The normal result CSVs store regret and timing, but they do not store the
selected `x_next` points. Because movement cost needs the selected points, use
the movement-cost calculator to re-run a best config and record movement per BO
iteration.

For one or more rewards from `output/reward_finetune_reward_params/<reward>/tuning/best_config.json`:

```bash
cd ~/my_implementation
source .venv/bin/activate

.venv/bin/python src/experiments/calculate_used_budget.py \
  --reward snake earlbo log_improvement normalized_improvement optimistic_improvement \
  --results-root output/reward_finetune_reward_params \
  --output-dir output/movement_cost_usage \
  --device cpu
```

Prefer PBS for the full standard-reward movement calculation:

```bash
qsub cluster/calculate_used_budget_standard.pbs
```

For explicit best-config paths:

```bash
.venv/bin/python src/experiments/calculate_used_budget.py \
  --best-config output/reward_finetune_reward_params/snake/tuning/best_config.json \
  --best-config output/reward_finetune_reward_params/earlbo/tuning/best_config.json \
  --output-dir output/movement_cost_usage \
  --device cpu
```

Outputs:

```text
src/movement_cost_usage/<reward>/movement_cost_by_iteration.csv
src/movement_cost_usage/<reward>/movement_cost_summary.json
src/movement_cost_usage/movement_cost_summary.json
```

For the dedicated budgeted-exploration tuning output:

```bash
.venv/bin/python src/experiments/calculate_used_budget.py \
  --reward budgeted_exploration \
  --results-root output/budgeted_exploration_budget_grid \
  --output-dir output/budgeted_exploration_movement_cost \
  --device cpu
```

Or submit it through PBS:

```bash
qsub cluster/calculate_used_budget_budgeted.pbs
```

## Better Parallel Version: One Config Per Array Job

The script above creates 5 jobs, one per reward, and each job loops through all
hyperparameter configs. This works, but it can queue for a long time and one
failure loses a lot of work.

The better PBS-array pattern is:

```text
array index -> one reward function + one hyperparameter config
```

With the current fast screening setup:

```text
earlbo: 4 configs
snake: 4 base configs x 3 path-cost weights = 12 configs
log_improvement: 4 base configs x 3 scale values = 12 configs
normalized_improvement: 4 configs
optimistic_improvement: 4 base configs x 3 std weights = 12 configs

total = 44 array jobs
```

The budget-aware reward uses:

```text
reward = improvement
       + remaining_budget_fraction * explore_weight * GP_std
       - (1 - remaining_budget_fraction) * path_cost_weight * move_cost
       - over_budget_penalty * over_budget
```

This encourages exploration while movement budget remains and shifts toward
shorter, exploitative moves as the budget is consumed. Its tuned parameters are:

```text
movement_budget
reward_param_explore_weight
reward_param_path_cost_weight
reward_param_over_budget_penalty
```

To tune only the budget-aware reward with a dedicated parameter grid, use:

```bash
qsub cluster/submit_budgeted_exploration_param_tuning.pbs
```

That tunes only the reward-specific budget parameters while keeping the base
PPO/search settings fixed. With the current compact grid it runs 1 config:

```text
movement_budget in [5]
reward_param_explore_weight in [5.0]
reward_param_path_cost_weight in [0.1]
reward_param_over_budget_penalty in [0.0]

1 config total
```

If the job fails, resubmit the same PBS script; completed configs are skipped
because the script uses `--skip-existing`.

If you want one PBS job per config instead, use:

```bash
qsub cluster/submit_budgeted_exploration_param_array.pbs
```

That submits array index `0`.

It writes to:

```text
output/budgeted_exploration_budget_grid/budgeted_exploration/
```

After it finishes, collect with:

```bash
.venv/bin/python src/experiments/run_budgeted_exploration_tuning.py --collect --output-root ../output/budgeted_exploration_budget_grid
```

The best config will be:

```text
output/budgeted_exploration_budget_grid/budgeted_exploration/tuning/best_config.json
```

## Dedicated Lookahead Budgeted Exploration Tuning

The path-aware lookahead budgeted reward is tuned separately from the standard
reward array and separately from `budgeted_exploration`.

Check the current number of lookahead configs:

```bash
.venv/bin/python src/experiments/run_lookahead_budgeted_exploration_tuning.py --print-total
```

Expected:

```text
9
```

The current compact grid is:

```text
movement_budget in [5]
reward_param_explore_weight in [5.0]
reward_param_path_cost_weight in [0.1]
reward_param_future_path_cost_weight in [0.01, 0.05, 0.1]
reward_param_future_optimism_weight in [0.5, 1.0, 2.0]
reward_param_future_softmax_temperature in [1.0]
reward_param_over_budget_penalty in [0.0]

9 configs total = 3 future path weights x 3 future optimism weights
```

To tune it with grouped PBS jobs:

```bash
qsub cluster/group_submit_lookahead_reward_finetune.pbs
```

That submits array indices `0-2`; each array job runs 3 configs.

If you want one PBS job per config instead, use:

```bash
qsub cluster/submit_lookahead_reward_config_array.pbs
```

That submits array indices `0-8`.

It writes to:

```text
output/lookahead_budgeted_exploration_budget_grid/lookahead_budgeted_exploration/
```

After it finishes, collect with:

```bash
.venv/bin/python src/experiments/run_lookahead_budgeted_exploration_tuning.py \
  --collect \
  --output-root ../output/lookahead_budgeted_exploration_budget_grid
```

The best config will be:

```text
output/lookahead_budgeted_exploration_budget_grid/lookahead_budgeted_exploration/tuning/best_config.json
```

To test the selected best config:

```bash
qsub cluster/submit_lookahead_reward_test.pbs
```

## Standard Reward Array

```bash
qsub cluster/submit_reward_config_array.pbs
```

This submits:

```text
0-3     -> earlbo configs
4-15    -> snake configs, including snake_path_cost_weight in [0.0, 0.01, 0.05]
16-27   -> log_improvement configs, including scale in [0.5, 1.0, 2.0]
28-31   -> normalized_improvement configs
32-43   -> optimistic_improvement configs, including std_weight in [0.1, 0.2, 0.5]
```

Each job writes one config result under:

```text
output/reward_finetune_reward_params/<reward>/tuning/config_<id>/
```

After all jobs finish, collect the best config per reward:

```bash
cd ~/my_implementation
source .venv/bin/activate
python src/experiments/run_reward_array.py --collect --output-root ../output/reward_finetune_reward_params
```

This writes:

```text
output/reward_finetune_reward_params/<reward>/tuning/tuning_results.csv
output/reward_finetune_reward_params/<reward>/tuning/best_config.json
output/reward_finetune_reward_params/reward_summary.json
```

For a smoke test, submit a small range first:

```bash
qsub -J 0-3 cluster/submit_reward_config_array.pbs
```

If those finish correctly, submit the full array:

```bash
qsub cluster/submit_reward_config_array.pbs
```
