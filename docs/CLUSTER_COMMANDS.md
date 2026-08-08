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

## Leave-One-Function-Out Tuning, Collection, And Testing

See `cluster/README.md` for the full leave-one-function-out workflow. The
short version, run from the cluster login node:

```bash
bash scripts/submit_leave_one_function_out_cluster.sh
```

This submits a tuning array, then a dependent collection array, then a
dependent held-out test array, one chain per default function/reward/dimension
set in `scripts/lofo_defaults.sh`. Override any of them with environment
variables, e.g.:

```bash
DIMENSIONS="2 10" REWARDS="earlbo snake" bash scripts/submit_leave_one_function_out_cluster.sh ackley levy
```

To run tuning, collection, or testing separately (e.g. to resubmit just one
stage after a failure), use `scripts/submit_lofo_tuning_cluster.sh`,
`scripts/submit_lofo_best_configs_cluster.sh`, and
`scripts/submit_lofo_tests_cluster.sh`.

To test the currently-defined singleton parameters (skip tuning) across every
function/reward pair:

```bash
qsub cluster/test_current_params_all_functions.pbs
```

## Check Produced File Counts Match Expected

Count how many `best_config.json` (tuning) and `test_config.json` (testing)
files exist versus how many are expected, for the current
`scripts/lofo_defaults.sh` defaults (or your overrides):

```bash
source scripts/lofo_defaults.sh
DIMENSION="${DIMENSION:-$LOFO_DEFAULT_DIMENSION}"
read -ra FUNCTIONS <<< "${FUNCTIONS:-$LOFO_DEFAULT_FUNCTIONS}"
read -ra REWARDS <<< "${REWARDS:-$LOFO_DEFAULT_REWARDS}"
RESULT_ROOT="output/leave_one_function_out/dimension_${DIMENSION}"

expected=$(( ${#FUNCTIONS[@]} * ${#REWARDS[@]} ))
best=0
tested=0
for f in "${FUNCTIONS[@]}"; do
  for r in "${REWARDS[@]}"; do
    root="$RESULT_ROOT/held_out_${f}/${r}"
    [ -f "$root/tuning/best_config.json" ] && best=$((best + 1))
    [ -f "$root/test_${f}/test_config.json" ] && tested=$((tested + 1))
  done
done
echo "best_configs=$best/$expected"
echo "test_configs=$tested/$expected"
```

## Sync Results Back To Local

```bash
rsync -av \
  bd225@login.cx3.hpc.imperial.ac.uk:~/my_implementation/output/leave_one_function_out/ \
  /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation/output/leave_one_function_out/
```

## Inspect Results

Show a fold's tuning table:

```bash
column -s, -t < output/leave_one_function_out/dimension_2/held_out_ackley/snake/tuning/leave_one_out_results.csv | less -S
```

Check a held-out test result:

```bash
python -m json.tool output/leave_one_function_out/dimension_2/held_out_ackley/snake/test_ackley/test_config.json
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
