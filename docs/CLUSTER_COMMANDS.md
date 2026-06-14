# Cluster Commands

Run local commands from the repository root on your Mac. Run cluster commands
from `~/my_implementation` on the cluster.

## Sync Code To Cluster

Local:

```bash
./scripts/sync_to_cluster.sh
```

Cluster:

```bash
ssh bd225@login.cx3.hpc.imperial.ac.uk
cd ~/my_implementation
```

## Check Job Status

Cluster:

```bash
qstat -u "$USER"
qstat -t -u "$USER"
```

Detailed status for one job:

```bash
qstat -xf '<JOB_ID>'
```

Delete a job:

```bash
qdel '<JOB_ID>'
```

## Standard Reward Tuning

Rewards:

```text
earlbo
snake
log_improvement
normalized_improvement
optimistic_improvement
```

Check total configs:

```bash
.venv/bin/python src/experiments/run_reward_array.py --print-total
```

Expected:

```text
44
```

Grouped tuning:

```bash
qsub cluster/group_submit_reward_finetune.pbs
```

One PBS array job per config:

```bash
qsub cluster/submit_reward_config_array.pbs
```

Collect:

```bash
.venv/bin/python src/experiments/run_reward_array.py \
  --collect \
  --output-root ../output/reward_finetune_reward_params
```

Test selected best configs:

```bash
qsub cluster/submit_reward_tests.pbs
```

Output:

```text
output/reward_finetune_reward_params/
```

## Budgeted Exploration Tuning

Reward:

```text
budgeted_exploration
```

Check total configs:

```bash
.venv/bin/python src/experiments/run_budgeted_exploration_tuning.py --print-total
```

Parallel tuning:

```bash
qsub cluster/submit_budgeted_exploration_param_tuning.pbs
```

With the current grid this submits 2 PBS array subjobs, one config per subjob,
and each config runs for 20 BO evaluations.

One PBS array job per config:

```bash
qsub -J 0-1 cluster/submit_budgeted_exploration_param_array.pbs
```

Collect:

```bash
.venv/bin/python src/experiments/run_budgeted_exploration_tuning.py \
  --collect \
  --output-root ../output/budgeted_exploration_budget_grid
```

Best config:

```bash
cat output/budgeted_exploration_budget_grid/budgeted_exploration/tuning/best_config.json
```

Output:

```text
output/budgeted_exploration_budget_grid/budgeted_exploration/
```

## Lookahead Budgeted Exploration Tuning

Reward:

```text
lookahead_budgeted_exploration
```

Check total configs:

```bash
.venv/bin/python src/experiments/run_lookahead_budgeted_exploration_tuning.py --print-total
```

Expected:

```text
9
```

Grouped tuning:

```bash
qsub cluster/group_submit_lookahead_reward_finetune.pbs
```

One PBS array job per config:

```bash
qsub cluster/submit_lookahead_reward_config_array.pbs
```

Collect:

```bash
.venv/bin/python src/experiments/run_lookahead_budgeted_exploration_tuning.py \
  --collect \
  --output-root ../output/lookahead_budgeted_exploration_budget_grid
```

Best config:

```bash
cat output/lookahead_budgeted_exploration_budget_grid/lookahead_budgeted_exploration/tuning/best_config.json
```

Test selected best config:

```bash
qsub cluster/submit_lookahead_reward_test.pbs
```

Output:

```text
output/lookahead_budgeted_exploration_budget_grid/lookahead_budgeted_exploration/
```

## Sync Results Back To Local

Local, standard rewards:

```bash
rsync -av \
  bd225@login.cx3.hpc.imperial.ac.uk:~/my_implementation/output/reward_finetune_reward_params/ \
  /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation/output/reward_finetune_reward_params/
```

Local, budgeted exploration:

```bash
rsync -av \
  bd225@login.cx3.hpc.imperial.ac.uk:~/my_implementation/output/budgeted_exploration_budget_grid/ \
  /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation/output/budgeted_exploration_budget_grid/
```

Local, lookahead budgeted exploration:

```bash
rsync -av \
  bd225@login.cx3.hpc.imperial.ac.uk:~/my_implementation/output/lookahead_budgeted_exploration_budget_grid/ \
  /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation/output/lookahead_budgeted_exploration_budget_grid/
```

## Inspect Results

Show tuning table:

```bash
column -s, -t < output/lookahead_budgeted_exploration_budget_grid/lookahead_budgeted_exploration/tuning/tuning_results.csv | less -S
```

Check final BO iteration in a run:

```bash
tail output/lookahead_budgeted_exploration_budget_grid/lookahead_budgeted_exploration/tuning/config_000/run_0000.csv
```

Check a config:

```bash
python -m json.tool output/lookahead_budgeted_exploration_budget_grid/lookahead_budgeted_exploration/tuning/config_000/config.json
```

## PBS Logs

Find logs for a job:

```bash
find . -maxdepth 1 -type f -name '*<JOB_ID>*' -print
```

Find likely output and error logs:

```bash
find . -maxdepth 1 -type f -name '*.o*' -print
find . -maxdepth 1 -type f -name '*.e*' -print
```

Read recent log output:

```bash
tail -100 '<LOG_FILE>'
```
