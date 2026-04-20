import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "main.py"


# Keep the experiment setup simple and editable in one place.
BASE_SETTINGS = {
    "dimension": 10,
    "test_func": "ackley",
    "horizon": 3,
    "lower_bound": -1.0,
    "upper_bound": 1.0,
    "num_initial_data": 30,
    "update_episode": 10,
    "no_improvement_threshold": 15,
    "ppo_action_std_min": 0.01,
    "ppo_k_epochs": 100,
    "ppo_eps_clip": 0.2,
    "ppo_gamma_increase": 1.0,
    "ppo_vf_coeff": 0.5,
    "ppo_freeze_num": 2,
    "gpr_rbf_length_scale": 1.0,
    "gpr_wk_noise_level": 1.0,
    "gpr_restarts": 10,
    "reward_mode": "earlbo",
    "snake_path_cost_weight": 0.0,
}

# TUNE_BUDGET = {"num_runs": 3, "num_experiments": 15}
TUNE_BUDGET = {"num_runs": 3, "num_experiments": 3}
TEST_BUDGET = {"num_runs": 10, "num_experiments": 30}

SEARCH_SPACE = {
    "max_episodes": [300],
    "off_policy_episodes": [40],
    "encoder_learning_rate": [1e-3, 1e-2],
    "ppo_learning_rate": [1e-4],
    "ppo_action_std": [0.1],
    "ppo_action_decay": [0.99],
    "ppo_gamma": [0.95],
    "ppo_entropy_coeff": [0.01],
    "gpr_alpha": [1e-8],
}

# [tune] base settings: reward_mode=earlbo, snake_path_cost_weight=0.0, device=cuda
# [tune] config_000 params: max_episodes=300, off_policy_episodes=40, encoder_learning_rate=0.001, ppo_learning_rate=0.0001, ppo_action_std=0.1, ppo_action_decay=0.99, ppo_gamma=0.95, ppo_entropy_coeff=0.01, gpr_alpha=1e-08
# Run 0 saved to /rds/general/user/bd225/home/my_implementation/EARL_BO/batch_runs/tuning/config_000/run_0000.csv
# Aggregated results saved to /rds/general/user/bd225/home/my_implementation/EARL_BO/batch_runs/tuning/config_000/RL_BO_10D_ackley_h3.csv

# NO
# ppo_learning_rate=0.0005
# off_policy_episodes=20

# [tune] config_000 params: max_episodes=300, off_policy_episodes=20, encoder_learning_rate=0.0005, ppo_learning_rate=0.0001, ppo_action_std=0.1, ppo_action_decay=0.99, ppo_gamma=0.95, ppo_entropy_coeff=0.01, gpr_alpha=1e-08
def parse_args():
    parser = argparse.ArgumentParser(description="Search EARL_BO hyperparameters and save the best result.")
    parser.add_argument("--mode", choices=("tune", "test", "both"), default="tune")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-root", default="batch_runs")
    parser.add_argument("--reward-mode", choices=("earlbo", "snake"), default="earlbo")
    parser.add_argument("--snake-path-cost-weight", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_sweep():
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def effective_base_settings(args):
    settings = dict(BASE_SETTINGS)
    if args.reward_mode is not None:
        settings["reward_mode"] = args.reward_mode
    if args.snake_path_cost_weight is not None:
        settings["snake_path_cost_weight"] = args.snake_path_cost_weight
    return settings


def base_command(output_dir, budget, device, args):
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--num-runs",
        str(budget["num_runs"]),
        "--num-experiments",
        str(budget["num_experiments"]),
    ]

    for key, value in effective_base_settings(args).items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    if device is not None:
        cmd.extend(["--device", device])
    return cmd


def config_flags(config):
    flags = []
    for key, value in config.items():
        flags.extend([f"--{key.replace('_', '-')}", str(value)])
    return flags


def summary_path(output_dir):
    return output_dir / (
        f"RL_BO_{BASE_SETTINGS['dimension']}D_{BASE_SETTINGS['test_func']}_h{BASE_SETTINGS['horizon']}.csv"
    )


def read_regrets(csv_path):
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [float(row["Avg Regret"]) for row in rows]


def score_result(csv_path):
    regrets = read_regrets(csv_path)
    tail = regrets[-min(5, len(regrets)):]
    return {
        "score": sum(tail) / len(tail),
        "final_regret": regrets[-1],
        "best_regret": min(regrets),
    }


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2))


def format_config(config):
    return ", ".join(f"{key}={value}" for key, value in config.items())


def copy_result(summary_csv, target_dir, metadata):
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_csv = target_dir / summary_csv.name
    shutil.copy2(summary_csv, copied_csv)
    save_json(target_dir / "metadata.json", metadata)
    return copied_csv


def run_one(config, run_dir, budget, device, skip_existing, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    result_csv = summary_path(run_dir)

    if skip_existing and result_csv.exists():
        return {"status": "skipped", **score_result(result_csv), "summary_csv": str(result_csv)}

    cmd = base_command(run_dir, budget, device, args) + config_flags(config)
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        return {"status": "failed", "summary_csv": str(result_csv)}
    if not result_csv.exists():
        return {"status": "missing_summary", "summary_csv": str(result_csv)}

    return {"status": "ok", **score_result(result_csv), "summary_csv": str(result_csv)}


def run_tuning(output_root, device, skip_existing, args):
    tuning_root = output_root / "tuning"
    rows = []
    base_settings = effective_base_settings(args)

    print(
        "[tune] base settings: "
        f"reward_mode={base_settings['reward_mode']}, "
        f"snake_path_cost_weight={base_settings['snake_path_cost_weight']}, "
        f"device={device or 'auto'}"
    )

    for index, config in enumerate(build_sweep()):
        run_dir = tuning_root / f"config_{index:03d}"
        print(f"[tune] config_{index:03d} params: {format_config(config)}")
        result = run_one(config, run_dir, TUNE_BUDGET, device, skip_existing, args)
        row = {"config_id": index, **config, **result}
        rows.append(row)
        print(f"[tune] config_{index:03d}: {result['status']}")

    tuning_root.mkdir(parents=True, exist_ok=True)
    results_csv = tuning_root / "tuning_results.csv"
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["status"] in ("ok", "skipped")]
    if not valid_rows:
        raise RuntimeError(f"No successful tuning runs. See {results_csv}")

    best_row = min(valid_rows, key=lambda row: (row["score"], row["final_regret"], row["best_regret"]))
    best_payload = {
        "config_id": best_row["config_id"],
        "params": {key: best_row[key] for key in SEARCH_SPACE},
        "score": best_row["score"],
        "final_regret": best_row["final_regret"],
        "best_regret": best_row["best_regret"],
        "summary_csv": best_row["summary_csv"],
    }
    saved_csv = copy_result(Path(best_row["summary_csv"]), tuning_root / "best_tuning", best_payload)
    best_payload["saved_summary_csv"] = str(saved_csv)
    save_json(tuning_root / "best_config.json", best_payload)
    return best_payload


def load_best(output_root):
    best_path = output_root / "tuning" / "best_config.json"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best config: {best_path}")
    return json.loads(best_path.read_text())


def run_test(output_root, best_payload, device, skip_existing, args):
    test_root = output_root / "test_best"
    result = run_one(best_payload["params"], test_root, TEST_BUDGET, device, skip_existing, args)
    payload = {
        "selected_from_config_id": best_payload["config_id"],
        "params": best_payload["params"],
        "test_result": result,
    }
    if result["status"] in ("ok", "skipped"):
        saved_csv = copy_result(Path(result["summary_csv"]), test_root / "saved_test_result", payload)
        payload["saved_summary_csv"] = str(saved_csv)
    save_json(test_root / "test_config.json", payload)
    return payload


def main():
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    best_payload = None
    if args.mode in ("tune", "both"):
        best_payload = run_tuning(output_root, args.device, args.skip_existing, args)
        print(f"Best config: {best_payload['params']}")

    if args.mode in ("test", "both"):
        if best_payload is None:
            best_payload = load_best(output_root)
        test_payload = run_test(output_root, best_payload, args.device, args.skip_existing, args)
        print(f"Test status: {test_payload['test_result']['status']}")


if __name__ == "__main__":
    main()
