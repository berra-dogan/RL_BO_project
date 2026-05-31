# Cluster PBS Scripts

Run these from the repository root on the cluster, for example:

```bash
cd ~/my_implementation
qsub cluster/submit_budgeted_exploration_param_tuning.pbs
```

## Reward Sweeps

- `submit_reward_config_array.pbs`: one PBS array job per non-budgeted reward/config.
- `group_submit_reward_finetune.pbs`: grouped non-budgeted reward sweep, 10 configs per PBS array job.
- `submit_reward_tests.pbs`: tests each reward's selected `best_config.json`.

## Budgeted Exploration

- `submit_budgeted_exploration_param_array.pbs`: one PBS array job per budgeted config.
- `submit_budgeted_exploration_param_tuning.pbs`: grouped budgeted sweep, 3 configs per PBS array job.

## Movement Cost

- `calculate_used_budget_standard.pbs`: reruns selected standard reward best configs and records movement cost.
- `calculate_used_budget_budgeted.pbs`: reruns the budgeted-exploration best config and records movement cost.
