"""One combined convergence figure comparing reward methods per objective function.

For a fixed (dimension, horizon) -- default 5D / horizon 3 -- this builds a single
figure with one row per objective function and two columns:

    left  column : best regret so far        (running min of per-iteration regret,
                                               vs. iteration, log y-axis)
    right column : best regret so far        (same regret series, but vs. test
                                               cumulative scaled movement cost;
                                               linear x-axis, log y-axis) -- shows
                                               which methods reach low regret for
                                               less movement cost spent, i.e. which
                                               use movement most efficiently

Both are the mean over the 3 held-out test runs with a +/-1 std shaded band.
Styling (Times font, bold function titles, dotted grid, one shared legend in a
rounded grey box centred at the bottom) mirrors report/figures/example_paper_graphs.png.

Methods drawn in every panel (7):

    Pure BO                                   output/pure_bo_grid/...
    Expected Improvement (EARL-BO)            output/earlbo_grid/...
    Expected Improvement with Movement Cost   snake
    Log Improvement with Movement Cost        log_improvement_movement_cost3
    Optimistic Improvement with Movement Cost optimistic_improvement_movement_cost2/_cost3,
                                             selected per fold from the winner column of
                                             src/summary/compare_optimistic_improvement_movement_cost2_vs_optimistic_improvement_movement_cost3.csv
                                             (override with --optimistic-variant {auto,2,3})
    Budgeted Exploration Improvement          budgeted_exploration
    Look-ahead Budgeted Exploration Improvement   lookahead_budgeted_exploration

RL rewards read from
    output/leave_one_function_out/dimension_<d>/horizon_<h>/held_out_<fn>/<reward>/test_<fn>/run_*.csv
earlbo / pure_bo read from
    output/<grid>/dimension_<d>/horizon_<h>/<fn>/run_*.csv

By default the 5 simple objectives (ackley, levy, rosenbrock, sphere, sum_square,
held-out folds) are used. With --complex, the 3 test-only complex objectives
(rastrigin, schwefel, michalewicz, read from all_functions/) are used instead.

Usage:
    python3 src/experiments/plot_reward_convergence.py                       # 5D / h3, simple
    python3 src/experiments/plot_reward_convergence.py --dimension 5 --horizon 3 --format pdf
    python3 src/experiments/plot_reward_convergence.py --all                 # every (dim, horizon)
    python3 src/experiments/plot_reward_convergence.py --all --complex       # complex objectives
    python3 src/experiments/plot_reward_convergence.py --dimensions 3 5 10 --horizons 3 5

One figure per (dimension, horizon) is written to (<kind> = simple | complex)
    report/figures/reward_convergence/<kind>/plots/<kind>_dim<d>_h<h>.<png|pdf>
    report/figures/reward_convergence/<kind>/data/<kind>_dim<d>_h<h>.csv

(Needs matplotlib -- use the miniconda `python3`, the repo venv does not have it.)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (
        LogFormatterSciNotation,
        LogLocator,
        NullFormatter,
    )
except ImportError:  # pragma: no cover - helpful message only
    sys.exit(
        "matplotlib is required. Run this with the miniconda python3 "
        "(`/opt/miniconda3/bin/python3`), which has it; the repo venv does not."
    )

ROOT = Path(__file__).resolve().parents[2]
LOFO = ROOT / "output" / "leave_one_function_out"
COMPARE_CSV = (
    ROOT
    / "src"
    / "summary"
    / "compare_optimistic_improvement_movement_cost2_vs_optimistic_improvement_movement_cost3.csv"
)

# the 5 tuning/"simple" objectives (LOFO held-out folds) and the 3 "complex"
# objectives that are test-only (evaluated under all_functions/, never held out).
SIMPLE_FUNCTIONS = ["ackley", "levy", "rosenbrock", "sphere", "sum_square"]
COMPLEX_FUNCTIONS = ["rastrigin", "schwefel", "michalewicz"]

FUNCTION_TITLES = {
    "ackley": "Ackley",
    "levy": "Levy",
    "rosenbrock": "Rosenbrock",
    "sphere": "Sphere",
    "sum_square": "Sum Squares",
    "michalewicz": "Michalewicz",
    "rastrigin": "Rastrigin",
    "schwefel": "Schwefel",
}

# (internal key, legend label, colour) -- order = legend order = draw order.
METHODS = [
    ("pure_bo", "Pure BO", "#7f7f7f"),
    ("earlbo", "Expected Improvement (EARL-BO)", "#d62728"),
    ("snake", "Expected Improvement with Movement Cost", "#1f77b4"),
    ("log_improvement_movement_cost3", "Log Improvement with Movement Cost", "#2ca02c"),
    ("optimistic_selected", "Optimistic Improvement with Movement Cost", "#9467bd"),
    ("budgeted_exploration", "Budgeted Exploration Improvement", "#ff7f0e"),
    ("lookahead_budgeted_exploration", "Look-ahead Budgeted Exploration Improvement", "#8c564b"),
]

MOVE_COST_COL = "Cumulative Scaled Move Cost"
REGRET_COL = "Regret"
ITER_COL = "Iteration"


def load_runs(run_dir: Path):
    """Return (iterations, best_regret[nruns, niter], cum_move_cost[nruns, niter])."""
    run_files = sorted(run_dir.glob("run_*.csv"))
    if not run_files:
        return None
    iters = None
    best_regret_rows = []
    move_cost_rows = []
    for rf in run_files:
        with rf.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        it = np.array([float(r[ITER_COL]) for r in rows])
        regret = np.array([float(r[REGRET_COL]) for r in rows])
        move_cost = np.array([float(r[MOVE_COST_COL]) for r in rows])
        if iters is None:
            iters = it
        n = min(len(iters), len(it))
        iters = iters[:n]
        best_regret_rows.append(np.minimum.accumulate(regret[:n]))
        move_cost_rows.append(move_cost[:n])
    n = min(len(r) for r in best_regret_rows)
    best_regret = np.vstack([r[:n] for r in best_regret_rows])
    move_cost = np.vstack([r[:n] for r in move_cost_rows])
    return iters[:n], best_regret, move_cost


def optimistic_variant_by_function(dimension: int, horizon: int, override: str):
    if override in ("2", "3"):
        return lambda fn: f"optimistic_improvement_movement_cost{override}"
    winners = {}
    if COMPARE_CSV.exists():
        with COMPARE_CSV.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if int(row["dimension"]) == dimension and int(row["horizon"]) == horizon:
                    w = row["winner"]
                    if w.startswith("optimistic_improvement_movement_cost"):
                        winners[row["held_out_function"]] = w
    return lambda fn: winners.get(fn, "optimistic_improvement_movement_cost2")


def run_dir_for(method, fn, dimension, horizon, opt_variant, *, complex_mode=False):
    d, h = f"dimension_{dimension}", f"horizon_{horizon}"
    # complex objectives are test-only: their runs live under all_functions/<reward>/,
    # not under a held_out_<fn>/ fold.
    fold_root = LOFO / d / h / ("all_functions" if complex_mode else f"held_out_{fn}")
    if method == "optimistic_selected":
        reward = opt_variant(fn)
        return fold_root / reward / f"test_{fn}", reward
    if method in ("snake", "log_improvement_movement_cost3", "budgeted_exploration",
                  "lookahead_budgeted_exploration"):
        return fold_root / method / f"test_{fn}", method
    if method == "earlbo":
        return ROOT / "output" / "earlbo_grid" / d / h / fn, "earlbo"
    if method == "pure_bo":
        return ROOT / "output" / "pure_bo_grid" / d / h / fn, "pure_bo"
    raise ValueError(method)


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.formatter.use_mathtext": True,
        }
    )


def draw_panel(ax, series, *, logy, ylabel, title, xlabel):
    means = []
    for key, label, colour in METHODS:
        if key not in series:
            continue
        xs, mean, std = series[key]
        means.append(mean)
        lo = mean - std
        if logy:
            # keep the shaded band off a log axis' floor
            lo = np.clip(lo, a_min=np.max(mean) * 1e-4, a_max=None)
        ax.plot(xs, mean, color=colour, linewidth=1.6, label=label)
        ax.fill_between(xs, lo, mean + std, color=colour, alpha=0.2, linewidth=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, color="0.65", alpha=0.7)
    ax.margins(x=0.02)

    if not means:
        return
    stacked = np.concatenate(means)

    if logy and np.all(stacked > 0):
        ax.set_yscale("log")
        lo, hi = float(stacked.min()), float(stacked.max())
        # focus the axis on the mean curves; bands may spill to the edges
        ax.set_ylim(lo * 0.6, hi * 1.7)
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        span_decades = np.log10(hi / lo)
        if span_decades <= 1.2:
            # small range: label the intermediate ticks like the reference figure
            ax.yaxis.set_minor_formatter(
                LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(10, 0.4))
            )
            ax.tick_params(axis="y", which="minor", labelsize=6)
        else:
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=20))
            ax.yaxis.set_minor_formatter(NullFormatter())
    elif not logy:
        ax.set_ylim(bottom=0)


def discover_dim_horizon_pairs():
    """Every (dimension, horizon) with a leave_one_function_out output tree."""
    pairs = []
    for d_dir in sorted(LOFO.glob("dimension_*")):
        try:
            d = int(d_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for h_dir in sorted(d_dir.glob("horizon_*")):
            try:
                h = int(h_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            pairs.append((d, h))
    return sorted(set(pairs))


def make_figure(dimension: int, horizon: int, args):
    label = "complex" if args.complex else "simple"
    plots_dir = args.outdir / label / "plots"
    data_dir = args.outdir / label / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    opt_variant = optimistic_variant_by_function(
        dimension, horizon, args.optimistic_variant
    )

    functions = args.functions
    nrows = len(functions)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=2, figsize=(11.0, 3.05 * nrows + 1.1), squeeze=False
    )

    dump_rows = []
    for r, fn in enumerate(functions):
        regret_series, move_series = {}, {}
        for key, _, _ in METHODS:
            run_dir, reward = run_dir_for(
                key, fn, dimension, horizon, opt_variant, complex_mode=args.complex
            )
            loaded = load_runs(run_dir)
            if loaded is None:
                print(f"  [skip] {fn:11s} {key:32s} no run_*.csv in {run_dir.relative_to(ROOT)}")
                continue
            iters, best_regret, move_cost = loaded
            regret_series[key] = (iters, best_regret.mean(0), best_regret.std(0))
            move_series[key] = (move_cost.mean(0), best_regret.mean(0), best_regret.std(0))
            print(f"  [ok]   {fn:11s} {key:32s} <- {reward} ({best_regret.shape[0]} runs)")
            for i, it in enumerate(iters):
                dump_rows.append(
                    {
                        "function": fn,
                        "method": key,
                        "reward_dir": reward,
                        "iteration": int(it),
                        "best_regret_mean": best_regret.mean(0)[i],
                        "best_regret_std": best_regret.std(0)[i],
                        "cum_scaled_move_cost_mean": move_cost.mean(0)[i],
                        "cum_scaled_move_cost_std": move_cost.std(0)[i],
                    }
                )

        obj = FUNCTION_TITLES.get(fn, fn.title())
        draw_panel(
            axes[r][0],
            regret_series,
            logy=True,
            ylabel="Regret",
            title=f"{obj}: Regret",
            xlabel="Iterations",
        )
        draw_panel(
            axes[r][1],
            move_series,
            logy=True,
            ylabel="Regret",
            title=f"{obj}: Regret vs. Movement Cost",
            xlabel="Cumulative Scaled Movement Cost",
        )

    handles = [
        plt.Line2D([], [], color=c, linewidth=3.0, label=lbl) for _, lbl, c in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor="0.92",
        edgecolor="0.6",
        fontsize=12,
        handlelength=2.2,
        columnspacing=1.3,
        borderpad=0.7,
        labelspacing=0.5,
        bbox_to_anchor=(0.5, 0.004),
    )
    legend_frac = 1.0 / fig.get_figheight()
    fig.tight_layout(rect=[0, legend_frac, 1, 1], h_pad=1.4)

    stem = f"{label}_dim{dimension}_h{horizon}"
    path = plots_dir / f"{stem}.{args.format}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")

    if dump_rows:
        dump_path = data_dir / f"{stem}.csv"
        with dump_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(dump_rows[0].keys()))
            writer.writeheader()
            writer.writerows(dump_rows)
        print(f"  wrote {dump_path.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dimension", type=int, default=5)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument(
        "--all",
        action="store_true",
        help="build a figure for every (dimension, horizon) found under "
        "output/leave_one_function_out/ instead of just --dimension/--horizon",
    )
    p.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        help="explicit list of dimensions (paired with --horizons as a full grid); "
        "overrides --dimension. Ignored when --all is set.",
    )
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        help="explicit list of horizons (paired with --dimensions as a full grid); "
        "overrides --horizon. Ignored when --all is set.",
    )
    p.add_argument(
        "--complex",
        action="store_true",
        help="use the 3 complex, test-only objectives (rastrigin, schwefel, "
        "michalewicz) read from all_functions/, and write under "
        "reward_convergence/complex/ with 'complex_' filenames "
        "(default: the 5 simple objectives under reward_convergence/simple/)",
    )
    p.add_argument(
        "--functions",
        nargs="+",
        default=None,
        help="override the objective-function list "
        "(default: simple set, or complex set with --complex)",
    )
    p.add_argument("--optimistic-variant", choices=["auto", "2", "3"], default="auto")
    p.add_argument(
        "--outdir", type=Path, default=ROOT / "report" / "figures" / "reward_convergence"
    )
    p.add_argument("--format", default="png", choices=["png", "pdf"])
    args = p.parse_args()

    if args.functions is None:
        args.functions = COMPLEX_FUNCTIONS if args.complex else SIMPLE_FUNCTIONS

    if args.all:
        pairs = discover_dim_horizon_pairs()
        if not pairs:
            sys.exit(f"--all: no dimension_*/horizon_* trees under {LOFO.relative_to(ROOT)}")
    elif args.dimensions or args.horizons:
        dims = args.dimensions or [args.dimension]
        hors = args.horizons or [args.horizon]
        pairs = [(d, h) for d in dims for h in hors]
    else:
        pairs = [(args.dimension, args.horizon)]

    style()
    print(f"Building {len(pairs)} figure(s): {pairs}")
    for dimension, horizon in pairs:
        print(f"== dimension {dimension}, horizon {horizon} ==")
        make_figure(dimension, horizon, args)


if __name__ == "__main__":
    main()
