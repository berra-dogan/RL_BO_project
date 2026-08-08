"""Tune on all benchmark functions except one, then test on the held-out function."""

import argparse
import csv
import datetime
import fcntl
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.experiment_runner import (
    BASE_SETTINGS,
    ROOT,
    TEST_BUDGET,
    run_one,
    save_json,
    tune_budget,
)
from experiments.reward_configs import (
    BUDGET_AWARE_REWARDS,
    BUDGET_SCORE_MOVE_WEIGHT,
    REWARD_NAMES,
    base_settings,
    current_reward_config,
    load_result,
    reward_configs,
    runner_args,
)


FUNCTION_NAMES = ("ackley", "sphere", "sum_square", "levy", "rosenbrock")


def append_warning(args, message, **context):
    """Append a machine-readable warning without racing parallel array jobs."""
    warning_path = fold_root(args) / "warnings.jsonl"
    warning_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "message": message,
        **context,
    }
    with warning_path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    print(f"[warning] {message}; recorded in {warning_path}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune rewards on all benchmark functions except --test-function, "
            "then evaluate on that held-out function."
        )
    )
    parser.add_argument("--test-function", required=True, choices=FUNCTION_NAMES)
    parser.add_argument(
        "--dimension",
        type=int,
        default=BASE_SETTINGS["dimension"],
        help=f"Objective dimension (default: {BASE_SETTINGS['dimension']}).",
    )
    parser.add_argument(
        "--reward",
        nargs="+",
        choices=REWARD_NAMES,
        default=list(REWARD_NAMES),
        help="Rewards to evaluate. Defaults to all rewards.",
    )
    parser.add_argument(
        "--mode",
        choices=("tune", "collect", "test", "all"),
        default="all",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run one flattened tuning job instead of the complete tuning sweep.",
    )
    parser.add_argument("--print-total", action="store_true")
    parser.add_argument("--output-root", default="../output/leave_one_function_out")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--use-current-params",
        action="store_true",
        help=(
            "During test mode, use the current singleton parameter spaces "
            "instead of loading a collected best_config.json."
        ),
    )
    return parser.parse_args()


def training_functions(test_function):
    return [name for name in FUNCTION_NAMES if name != test_function]


def fold_root(args):
    return (
        ROOT
        / args.output_root
        / f"dimension_{args.dimension}"
        / f"held_out_{args.test_function}"
    ).resolve()


def build_jobs(rewards, test_function):
    return [
        (reward, config_id, config, function_name)
        for reward in rewards
        for config_id, config in enumerate(reward_configs(reward))
        for function_name in training_functions(test_function)
    ]


def function_runner_args(reward, function_name, settings=None):
    args = runner_args(reward, settings)
    args.test_func = function_name
    args.dimension = settings.get("dimension") if settings else None
    return args


def run_tuning_job(args, job):
    reward, config_id, config, function_name = job
    run_dir = (
        fold_root(args)
        / reward
        / "tuning"
        / f"config_{config_id:03d}"
        / function_name
    )
    result_path = run_dir / "result.json"
    config_path = run_dir / "config.json"
    budget = tune_budget(args.dimension)
    expected = {
        "reward": reward,
        "config_id": config_id,
        "training_function": function_name,
        "held_out_function": args.test_function,
        "params": config,
        "budget": budget,
    }
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and result_path.exists():
        if config_path.exists() and json.loads(config_path.read_text()) == expected:
            print(f"Result exists, skipping: {result_path}")
            return
        raise RuntimeError(f"Stale or mismatched result directory: {run_dir}")

    save_json(config_path, expected)
    print(f"reward={reward} config={config_id} function={function_name}")
    result = run_one(
        config,
        run_dir,
        budget,
        args.device,
        args.skip_existing,
        function_runner_args(
            reward,
            function_name,
            {"dimension": args.dimension},
        ),
    )
    save_json(result_path, result)
    if result["status"] not in ("ok", "skipped"):
        append_warning(
            args,
            "Tuning job did not complete successfully",
            reward=reward,
            config_id=config_id,
            training_function=function_name,
            held_out_function=args.test_function,
            dimension=args.dimension,
            status=result["status"],
            result=result,
        )


def tune(args):
    jobs = build_jobs(args.reward, args.test_function)
    if args.index is not None:
        if args.index < 0 or args.index >= len(jobs):
            raise IndexError(f"Index {args.index} outside valid range 0-{len(jobs) - 1}")
        job = jobs[args.index]
        try:
            run_tuning_job(args, job)
        except Exception as exc:
            reward, config_id, _, function_name = job
            append_warning(
                args,
                "Unhandled tuning-job exception",
                reward=reward,
                config_id=config_id,
                training_function=function_name,
                held_out_function=args.test_function,
                dimension=args.dimension,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        return
    for job in jobs:
        try:
            run_tuning_job(args, job)
        except Exception as exc:
            reward, config_id, _, function_name = job
            append_warning(
                args,
                "Unhandled tuning-job exception",
                reward=reward,
                config_id=config_id,
                training_function=function_name,
                held_out_function=args.test_function,
                dimension=args.dimension,
                error_type=type(exc).__name__,
                error=str(exc),
            )


def average_ranks(values):
    """Return one-based average ranks, assigning equal values equal ranks."""
    ranks = {}
    sorted_values = sorted(enumerate(values), key=lambda item: item[1])
    position = 0
    while position < len(sorted_values):
        end = position + 1
        while end < len(sorted_values) and sorted_values[end][1] == sorted_values[position][1]:
            end += 1
        rank = ((position + 1) + end) / 2
        for original_index, _ in sorted_values[position:end]:
            ranks[original_index] = rank
        position = end
    return [ranks[index] for index in range(len(values))]


def selection_score(reward, result):
    if reward not in BUDGET_AWARE_REWARDS:
        return result["score"]
    move_cost = result.get("mean_total_scaled_move_cost")
    return result["score"] + (
        BUDGET_SCORE_MOVE_WEIGHT * move_cost if move_cost is not None else 0.0
    )


def collect_reward(args, reward):
    functions = training_functions(args.test_function)
    configs = reward_configs(reward)
    rows = [{"config_id": index, **config} for index, config in enumerate(configs)]

    for function_name in functions:
        valid_indices = []
        scores = []
        for config_id, row in enumerate(rows):
            run_dir = (
                fold_root(args)
                / reward
                / "tuning"
                / f"config_{config_id:03d}"
                / function_name
            )
            result = load_result(run_dir)
            row[f"{function_name}_status"] = result["status"]
            if result["status"] in ("ok", "skipped"):
                score = selection_score(reward, result)
                row[f"{function_name}_score"] = score
                valid_indices.append(config_id)
                scores.append(score)

        for config_id, rank in zip(valid_indices, average_ranks(scores)):
            rows[config_id][f"{function_name}_rank"] = rank

    for row in rows:
        ranks = [row.get(f"{name}_rank") for name in functions]
        row["mean_rank"] = (
            sum(ranks) / len(ranks) if all(rank is not None for rank in ranks) else None
        )

    tuning_root = fold_root(args) / reward / "tuning"
    tuning_root.mkdir(parents=True, exist_ok=True)
    results_csv = tuning_root / "leave_one_out_results.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["mean_rank"] is not None]
    if not valid_rows:
        append_warning(
            args,
            "No complete configuration was available during collection",
            reward=reward,
            held_out_function=args.test_function,
            dimension=args.dimension,
            results_csv=str(results_csv),
        )
        return None

    incomplete_count = len(rows) - len(valid_rows)
    if incomplete_count:
        append_warning(
            args,
            "Collection excluded incomplete configurations",
            reward=reward,
            held_out_function=args.test_function,
            dimension=args.dimension,
            incomplete_configurations=incomplete_count,
            complete_configurations=len(valid_rows),
            results_csv=str(results_csv),
        )

    best = min(valid_rows, key=lambda row: (row["mean_rank"], row["config_id"]))
    selected_base_settings = base_settings(reward, configs[best["config_id"]])
    selected_base_settings["dimension"] = args.dimension
    selected_base_settings["test_func"] = args.test_function
    payload = {
        "reward": reward,
        "held_out_function": args.test_function,
        "training_functions": functions,
        "selection_metric": "mean_within_function_rank",
        "config_id": best["config_id"],
        "mean_rank": best["mean_rank"],
        "function_scores": {
            name: best[f"{name}_score"] for name in functions
        },
        "function_ranks": {
            name: best[f"{name}_rank"] for name in functions
        },
        "params": configs[best["config_id"]],
        "base_settings": selected_base_settings,
    }
    save_json(tuning_root / "best_config.json", payload)
    return payload


def collect(args):
    summary = {}
    for reward in args.reward:
        result = collect_reward(args, reward)
        if result is not None:
            summary[reward] = result
    save_json(fold_root(args) / "reward_summary.json", summary)
    print(f"Saved fold summary to {fold_root(args) / 'reward_summary.json'}")


def test(args):
    failures = []
    for reward in args.reward:
        reward_root = fold_root(args) / reward
        best_path = reward_root / "tuning" / "best_config.json"
        if args.use_current_params:
            params = current_reward_config(reward)
            selected_settings = base_settings(reward, params)
            selected_settings["dimension"] = args.dimension
            selected_settings["test_func"] = args.test_function
            best = {
                "config_id": None,
                "mean_rank": None,
                "params": params,
                "base_settings": selected_settings,
            }
            parameter_source = "current_parameter_spaces"
        else:
            if not best_path.exists():
                failures.append(f"{reward}: missing {best_path}")
                continue
            best = json.loads(best_path.read_text())
            parameter_source = str(best_path)

        test_root = reward_root / f"test_{args.test_function}"
        result = run_one(
            best["params"],
            test_root,
            TEST_BUDGET,
            args.device,
            args.skip_existing,
            function_runner_args(
                reward,
                args.test_function,
                best.get("base_settings"),
            ),
        )
        payload = {
            "reward": reward,
            "held_out_function": args.test_function,
            "parameter_source": parameter_source,
            "selected_from_config_id": best["config_id"],
            "selection_mean_rank": best["mean_rank"],
            "params": best["params"],
            "test_result": result,
            "used_budget": {
                key: value
                for key, value in result.items()
                if key.startswith("mean_") or key.startswith("std_total_")
            },
        }
        save_json(test_root / "test_config.json", payload)
        print(f"{reward} on {args.test_function}: {result['status']}")
        if result["status"] not in ("ok", "skipped"):
            failures.append(f"{reward}: {result['status']}")

    if failures:
        raise RuntimeError("Test failures:\n" + "\n".join(failures))


def main():
    args = parse_args()
    jobs = build_jobs(args.reward, args.test_function)
    if args.print_total:
        print(len(jobs))
        return
    if args.index is not None and args.mode != "tune":
        raise ValueError("--index can only be used with --mode tune")
    if args.mode in ("tune", "all"):
        tune(args)
    if args.mode in ("collect", "all"):
        collect(args)
    if args.mode in ("test", "all"):
        test(args)


if __name__ == "__main__":
    main()
