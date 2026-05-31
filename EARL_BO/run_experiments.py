import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

from rewards import available_reward_modes


ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT / "main.py"


BASE_SETTINGS = {
    "dimension": 10,
    "test_func": "ackley",
    "horizon": 3,
    "lower_bound": -1.0,
    "upper_bound": 1.0,
    "num_initial_data": 30,
    "update_episode": 10,
    "no_improvement_threshold": 8,
    "ppo_action_std_min": 0.01,
    "ppo_k_epochs": 30,
    "ppo_eps_clip": 0.2,
    "ppo_gamma_increase": 1.0,
    "ppo_vf_coeff": 0.5,
    "ppo_freeze_num": 2,
    "gpr_rbf_length_scale": 1.0,
    "gpr_wk_noise_level": 1.0,
    "gpr_restarts": 2,
    "snake_path_cost_weight": 0.0,
}

TUNE_BUDGET = {"num_runs": 1, "num_experiments": 5}
TEST_BUDGET = {"num_runs": 3, "num_experiments": 1}

SEARCH_SPACE = {
    # Keep reward_mode outside this grid when using PBS arrays:
    # each array index should run one reward function.
    #
    # Reward-specific parameters can still be swept here if needed:
    # "snake_path_cost_weight": [0.0, 0.01, 0.05],
    # "reward_param_std_weight": [0.1, 0.2, 0.5],
    # "reward_param_scale": [0.5, 1.0, 2.0],
    # "movement_budget": [0.5, 1.0, 2.0],
    # "reward_param_explore_weight": [0.1, 0.2, 0.5],
    # "reward_param_path_cost_weight": [0.01, 0.05, 0.1],
    "max_episodes": [120],
    "off_policy_episodes": [20],
    "encoder_learning_rate": [1e-3],
    "ppo_learning_rate": [1e-4, 2e-4],
    "ppo_action_std": [0.05, 0.1],
    "ppo_action_decay": [0.99],
    "ppo_gamma": [0.95],
    "ppo_entropy_coeff": [0.01],
    "gpr_alpha": [1e-8],
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Search EARL_BO hyperparameters and save the best result."
    )
    parser.add_argument("--mode", choices=("tune", "test", "both"), default="tune")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-root", default="batch_runs")
    parser.add_argument("--reward-mode", choices=available_reward_modes(), default="snake")
    parser.add_argument("--snake-path-cost-weight", type=float, default=None)
    parser.add_argument("--movement-budget", type=float, default=None)
    parser.add_argument(
        "--reward-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Reward-specific numeric parameter forwarded to main.py. Can be repeated.",
    )
    parser.add_argument(
        "--reward-params-json",
        default=None,
        help="JSON object of reward-specific numeric parameters forwarded to main.py.",
    )
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
    if args.movement_budget is not None:
        settings["movement_budget"] = args.movement_budget
    return settings


def summary_filename(settings):
    return (
        f"RL_BO_{settings['dimension']}D_"
        f"{settings['test_func']}_h{settings['horizon']}.csv"
    )


def summary_path(output_dir, args):
    settings = effective_base_settings(args)
    return output_dir / summary_filename(settings)


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
    for reward_param in args.reward_param:
        cmd.extend(["--reward-param", reward_param])
    if args.reward_params_json is not None:
        cmd.extend(["--reward-params-json", args.reward_params_json])

    return cmd


def config_flags(config):
    flags = []
    for key, value in config.items():
        if key == "reward_params":
            flags.extend(["--reward-params-json", json.dumps(value, sort_keys=True)])
            continue
        if key.startswith("reward_param_"):
            flags.extend(["--reward-param", f"{key.removeprefix('reward_param_')}={value}"])
            continue
        flags.extend([f"--{key.replace('_', '-')}", str(value)])
    return flags


def read_regrets(csv_path):
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows found in summary CSV: {csv_path}")

    regrets = []
    for i, row in enumerate(rows):
        if "Avg Regret" not in row:
            raise KeyError(f"'Avg Regret' column missing in row {i} of {csv_path}")
        regrets.append(float(row["Avg Regret"]))

    return regrets


def score_result(csv_path):
    regrets = read_regrets(csv_path)
    tail = regrets[-min(5, len(regrets)):]
    return {
        "score": sum(tail) / len(tail),
        "final_regret": regrets[-1],
        "best_regret": min(regrets),
    }


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def format_config(config):
    return ", ".join(f"{key}={value}" for key, value in config.items())


def copy_file_result(summary_csv, target_dir, metadata):
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_csv = target_dir / summary_csv.name
    shutil.copy2(summary_csv, copied_csv)
    save_json(target_dir / "metadata.json", metadata)
    return copied_csv


def copy_run_snapshot(run_dir, target_dir, metadata):
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(run_dir, target_dir)
    save_json(target_dir / "metadata.json", metadata)
    return target_dir


def run_one(config, run_dir, budget, device, skip_existing, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    result_csv = summary_path(run_dir, args)

    if skip_existing and result_csv.exists():
        try:
            metrics = score_result(result_csv)
            return {
                "status": "skipped",
                **metrics,
                "summary_csv": str(result_csv),
                "run_dir": str(run_dir),
            }
        except Exception as exc:
            print(f"[warn] Existing result invalid at {result_csv}: {exc}. Re-running.")

    cmd = base_command(run_dir, budget, device, args) + config_flags(config)
    completed = subprocess.run(cmd, cwd=ROOT, check=False)

    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "summary_csv": str(result_csv),
            "run_dir": str(run_dir),
        }

    if not result_csv.exists():
        return {
            "status": "missing_summary",
            "summary_csv": str(result_csv),
            "run_dir": str(run_dir),
        }

    try:
        metrics = score_result(result_csv)
    except Exception as exc:
        return {
            "status": "invalid_summary",
            "error": str(exc),
            "summary_csv": str(result_csv),
            "run_dir": str(run_dir),
        }

    return {
        "status": "ok",
        **metrics,
        "summary_csv": str(result_csv),
        "run_dir": str(run_dir),
    }


def run_tuning(output_root, device, skip_existing, args):
    tuning_root = output_root / "tuning"
    rows = []
    base_settings = effective_base_settings(args)
    sweep = build_sweep()

    if not sweep:
        raise RuntimeError("SEARCH_SPACE produced no configurations.")

    print(
        "[tune] base settings: "
        f"reward_mode={base_settings['reward_mode']}, "
        f"snake_path_cost_weight={base_settings['snake_path_cost_weight']}, "
        f"device={device or 'auto'}"
    )

    for index, config in enumerate(sweep):
        run_dir = tuning_root / f"config_{index:03d}"
        print(f"[tune] config_{index:03d} params: {format_config(config)}")
        result = run_one(config, run_dir, TUNE_BUDGET, device, skip_existing, args)
        row = {"config_id": index, **config, **result}
        rows.append(row)
        print(f"[tune] config_{index:03d}: {result['status']}")

    tuning_root.mkdir(parents=True, exist_ok=True)
    results_csv = tuning_root / "tuning_results.csv"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["status"] in ("ok", "skipped")]
    if not valid_rows:
        raise RuntimeError(f"No successful tuning runs. See {results_csv}")

    best_row = min(
        valid_rows,
        key=lambda row: (row["score"], row["final_regret"], row["best_regret"]),
    )

    best_payload = {
        "config_id": best_row["config_id"],
        "params": {key: best_row[key] for key in SEARCH_SPACE},
        "base_settings": base_settings,
        "score": best_row["score"],
        "final_regret": best_row["final_regret"],
        "best_regret": best_row["best_regret"],
        "summary_csv": best_row["summary_csv"],
        "run_dir": best_row["run_dir"],
    }

    saved_snapshot = copy_run_snapshot(
        Path(best_row["run_dir"]),
        tuning_root / "best_tuning",
        best_payload,
    )
    best_payload["saved_run_dir"] = str(saved_snapshot)
    best_payload["saved_summary_csv"] = str(saved_snapshot / Path(best_row["summary_csv"]).name)

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
        "base_settings": effective_base_settings(args),
        "test_result": result,
    }

    if result["status"] in ("ok", "skipped"):
        saved_snapshot = copy_run_snapshot(
            Path(result["run_dir"]),
            test_root / "saved_test_result",
            payload,
        )
        payload["saved_run_dir"] = str(saved_snapshot)
        payload["saved_summary_csv"] = str(saved_snapshot / Path(result["summary_csv"]).name)

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
