# Fine Tune Each Reward Function With PBS Array Jobs

This workflow uses the PBS array-job pattern from `HPC Array Jobs_ An Intro.pdf`.
Each array element fine-tunes one reward function by running `EARL_BO/run_experiments.py`
with a separate output directory.

## 1. Confirm Reward Functions

Reward functions are registered in `EARL_BO/rewards.py`.

Current reward modes:

```text
earlbo
snake
log_improvement
normalized_improvement
optimistic_improvement
```

To add another reward:

1. Add a function in `EARL_BO/rewards.py`.
2. Add it to `REWARD_FUNCTIONS`.
3. The new name will automatically appear in `--reward-mode`.

## 2. Choose The Hyperparameter Search Space

Edit `SEARCH_SPACE` in `EARL_BO/run_experiments.py`.

Example:

```python
SEARCH_SPACE = {
    "max_episodes": [300],
    "off_policy_episodes": [20, 40],
    "encoder_learning_rate": [1e-3, 1e-2],
    "ppo_learning_rate": [1e-4, 2e-4],
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

## 4. Create A PBS Array Script

On the cluster, create `submit_reward_finetune.pbs` inside `~/earl_bo_project`:

```bash
#!/bin/bash
#PBS -N reward_finetune
#PBS -J 0-4
#PBS -l select=1:ncpus=10:mem=32gb
#PBS -l walltime=08:00:00

cd "$PBS_O_WORKDIR"

module purge
module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10

mkdir -p logs

REWARDS=(earlbo snake log_improvement normalized_improvement optimistic_improvement)
REWARD="${REWARDS[$PBS_ARRAY_INDEX]}"

EXTRA_ARGS=""

if [ "$REWARD" = "snake" ]; then
  EXTRA_ARGS="--snake-path-cost-weight 0.01"
fi

if [ "$REWARD" = "optimistic_improvement" ]; then
  EXTRA_ARGS="--reward-param std_weight=0.2"
fi

echo "Array index: $PBS_ARRAY_INDEX"
echo "Reward mode: $REWARD"
echo "Extra args: $EXTRA_ARGS"

python EARL_BO/run_experiments.py \
  --mode tune \
  --device cpu \
  --output-root "reward_finetune/${REWARD}" \
  --reward-mode "$REWARD" \
  $EXTRA_ARGS \
  --skip-existing
```

Because there are 5 reward modes, the array range is `#PBS -J 0-4`.
If you add or remove reward modes, update both the `REWARDS=(...)` list and
the PBS range.

## 5. Submit A Small Test First

From the cluster login node:

```bash
cd ~/earl_bo_project
qsub -J 0-0 submit_reward_finetune.pbs
```

Monitor it:

```bash
qstat -u $USER -t
```

After it finishes, check output:

```bash
find EARL_BO/reward_finetune -name best_config.json
```

If the test works, submit all rewards:

```bash
qsub submit_reward_finetune.pbs
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
qsub submit_reward_finetune.pbs
```

Check completed reward searches:

```bash
find EARL_BO/reward_finetune -path "*/tuning/best_config.json" -print
```

Expected outputs:

```text
EARL_BO/reward_finetune/earlbo/tuning/best_config.json
EARL_BO/reward_finetune/snake/tuning/best_config.json
EARL_BO/reward_finetune/log_improvement/tuning/best_config.json
EARL_BO/reward_finetune/normalized_improvement/tuning/best_config.json
EARL_BO/reward_finetune/optimistic_improvement/tuning/best_config.json
```

## 8. Compare The Best Reward Configs

Each reward writes:

```text
EARL_BO/reward_finetune/<reward>/tuning/best_config.json
EARL_BO/reward_finetune/<reward>/tuning/tuning_results.csv
EARL_BO/reward_finetune/<reward>/tuning/best_tuning/
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
python EARL_BO/run_experiments.py \
  --mode test \
  --device cpu \
  --output-root "reward_finetune/${REWARD}" \
  --reward-mode "$REWARD" \
  $EXTRA_ARGS \
  --skip-existing
```

This loads:

```text
EARL_BO/reward_finetune/<reward>/tuning/best_config.json
```

and writes the final result under:

```text
EARL_BO/reward_finetune/<reward>/test_best/
```

## 10. Copy Results Back To Your Mac

Run this on your Mac:

```bash
cd /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation
rsync -av bd225@login.cx3.hpc.imperial.ac.uk:~/my_implementation/EARL_BO/reward_finetune/ EARL_BO/reward_finetune/
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

Use this PBS script instead:

```bash
qsub submit_reward_config_array.pbs
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
EARL_BO/reward_finetune_reward_params/<reward>/tuning/config_<id>/
```

After all jobs finish, collect the best config per reward:

```bash
cd ~/earl_bo_project
source .venv/bin/activate
python EARL_BO/run_reward_array.py --collect --output-root reward_finetune_reward_params
```

This writes:

```text
EARL_BO/reward_finetune_reward_params/<reward>/tuning/tuning_results.csv
EARL_BO/reward_finetune_reward_params/<reward>/tuning/best_config.json
EARL_BO/reward_finetune_reward_params/reward_summary.json
```

For a smoke test, submit a small range first:

```bash
qsub -J 0-3 submit_reward_config_array.pbs
```

If those finish correctly, submit the full array:

```bash
qsub submit_reward_config_array.pbs
```
