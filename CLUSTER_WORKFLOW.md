# Cluster Workflow

This workflow matches the exact commands for your thesis runs.

- `Local Mac path`: `/Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation`
- `Cluster path`: `~/earl_bo_project`

Only run cluster scripts after SSH login. Do not run these on your Mac:

- `scripts/setup_cluster_env.sh`
- `scripts/start_cluster_session.sh`
- `scripts/check_gpu.sh`
- `scripts/run_earl_bo.sh`
- `scripts/run_aggregate.sh`

## 1. Sync Local Code To The Cluster

Run this on your Mac if you changed the code:

```bash
cd /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation
./scripts/sync_to_cluster.sh
```

That copies the project to:

```bash
~/earl_bo_project
```

on the cluster.

Equivalent command:

```bash
rsync -av \
  --exclude '.venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation/ \
  bd225@login.cx3.hpc.imperial.ac.uk:~/earl_bo_project/
```

## 2. SSH To The Cluster

Run this on your Mac:

```bash
ssh bd225@login.cx3.hpc.imperial.ac.uk
```

## 3. Get A GPU

Run this on the cluster login node:

```bash
qsub -I -l select=1:ncpus=10:mem=32gb:ngpus=1 -l walltime=02:00:00
```

Or use the helper:

```bash
cd ~/earl_bo_project
bash scripts/start_cluster_session.sh
```

When `qsub -I` places you on a compute node, run the same helper again to load the module, activate `.venv`, and check CUDA:

```bash
cd ~/earl_bo_project
bash scripts/start_cluster_session.sh
```

## 4. Load Python And Activate The Virtual Environment

Run this on the cluster compute node:

```bash
module purge
module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
cd ~/earl_bo_project
source .venv/bin/activate
```

## 5. Check GPU

Run this on the cluster compute node:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

For a fuller diagnostic:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu'); print(torch.version.cuda)"
```

## 6. Run The Code

Run this on the cluster:

```bash
cd ~/earl_bo_project
source .venv/bin/activate
cd EARL_BO
python main.py
```

Or use the helper:

```bash
cd ~/earl_bo_project
bash scripts/run_earl_bo.sh
```

## 7. Aggregate Previously Run Outputs

Run this on the cluster:

```bash
cd ~/earl_bo_project
source .venv/bin/activate
cd EARL_BO
python main.py --aggregate-only --output-dir results
```

Or use the helper:

```bash
cd ~/earl_bo_project
bash scripts/run_aggregate.sh
```

## 8. Set Up Python On The Cluster

Run this on the cluster if the environment does not exist yet:

```bash
cd ~/earl_bo_project
bash scripts/setup_cluster_env.sh
```

Equivalent manual steps:

```bash
module purge
module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0
python --version
which python
cd ~/earl_bo_project
python -m venv .venv
source .venv/bin/activate
python --version
which python
pip install -r requirements.txt
```

If you need to recreate the environment from scratch:

```bash
cd ~/earl_bo_project
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 9. Copy Outputs Back To Your Mac

Run this on your Mac:

```bash
cd /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation
./scripts/fetch_results.sh
```

Or run `scp` directly:

```bash
scp bd225@login.cx3.hpc.imperial.ac.uk:~/earl_bo_project/EARL_BO/results/RL_BO_30D_ackley_h3.csv /Users/berra.dogan/Desktop/Imperial-Coursework/thesis/my_implementation/EARL_BO/
```
