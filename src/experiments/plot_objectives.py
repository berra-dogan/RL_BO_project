"""Plot the benchmark objective landscapes defined in objective_functions.py.

Each objective is evaluated at a fixed dimension N (default 3, i.e. the
`dimension_3` setting used across the LOFO / earlbo / pure_bo pipelines) and
rendered as a 3D surface: two axes (--axes, default the first two, x0/x1) are
swept over the function's search domain (its CANONICAL_DOMAINS entry, or
[-1, 1] otherwise) while every other axis is held fixed at the domain
midpoint. For N == 1 there is nothing to hold fixed, so a 2D line plot is
drawn instead.

By default the 5 simple objectives (ackley, sphere, sum_square, levy,
rosenbrock) are plotted as one combined figure: a 2x2 grid with the 5th
(last) objective centered on its own row underneath. Pass --complex to also
plot the 3 complex objectives (rastrigin, schwefel, michalewicz) as a second
combined figure: one row of 2, with the 3rd (last) objective centered on the
row underneath. This 2-per-row / last-one-centered layout is applied
generically, so an explicit --functions list is laid out the same way.

Usage:
    python3 src/experiments/plot_objectives.py                    # simple.png: 2x2 + 1 centered
    python3 src/experiments/plot_objectives.py --complex           # + complex.png: 1x2 + 1 centered
    python3 src/experiments/plot_objectives.py --dimension 5       # 5D objectives, x0/x1 swept, rest fixed
    python3 src/experiments/plot_objectives.py --dimension 1       # 2D line plots
    python3 src/experiments/plot_objectives.py --functions ackley rastrigin

Figures are written to report/objectives/<simple|complex|objectives>_dim<N>.<png|pdf>.

(Needs matplotlib -- use the miniconda python3, the repo venv does not have it.)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the '3d' projection)
except ImportError:  # pragma: no cover - helpful message only
    sys.exit(
        "matplotlib is required. Run this with the miniconda python3 "
        "(`/opt/miniconda3/bin/python3`), which has it; the repo venv does not."
    )

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from objective_functions import ObjectiveFunctions, search_domain  # noqa: E402

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

DEFAULT_DOMAIN = (-1.0, 1.0)


def domain_for(name: str):
    return search_domain(name) or DEFAULT_DOMAIN


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
        }
    )


def evaluate_grid(name: str, dimension: int, axes, resolution: int):
    """Sweep `axes` (1 or 2 indices) over the function's domain, other axes
    fixed at the domain midpoint. Returns the swept coordinate array(s) and
    the evaluated grid."""
    lo, hi = domain_for(name)
    mid = (lo + hi) / 2.0
    fixed = np.full(dimension, mid)
    fn = ObjectiveFunctions(dimension)

    if len(axes) == 1:
        (a0,) = axes
        xs = np.linspace(lo, hi, resolution)
        points = np.tile(fixed, (resolution, 1))
        points[:, a0] = xs
        zs = fn.evaluate(name, points)
        return xs, zs

    a0, a1 = axes
    xs = np.linspace(lo, hi, resolution)
    ys = np.linspace(lo, hi, resolution)
    X, Y = np.meshgrid(xs, ys)
    points = np.tile(fixed, (resolution * resolution, 1))
    points[:, a0] = X.ravel()
    points[:, a1] = Y.ravel()
    Z = fn.evaluate(name, points).reshape(resolution, resolution)
    return X, Y, Z


def grid_positions(n: int):
    """(nrows, [(row, col_start, col_end), ...]) laying out `n` items 2 per
    row (each spanning half of a 4-wide virtual grid), with a trailing
    odd item centered on its own row (spanning the middle 2 of 4 columns)."""
    nrows = (n + 1) // 2
    positions = []
    for i in range(n):
        row = i // 2
        is_last_alone = (i == n - 1) and (n % 2 == 1)
        if is_last_alone:
            positions.append((row, 1, 3))
        elif i % 2 == 0:
            positions.append((row, 0, 2))
        else:
            positions.append((row, 2, 4))
    return nrows, positions


def draw_objective(ax, name: str, dimension: int, axes, resolution: int):
    title = FUNCTION_TITLES.get(name, name.title())
    if len(axes) == 1:
        (a0,) = axes
        xs, zs = evaluate_grid(name, dimension, axes, resolution)
        ax.plot(xs, zs, color="#1f77b4", linewidth=1.8)
        ax.set_xlabel(f"x{a0}")
        ax.set_ylabel("f(x)")
        ax.grid(True, linestyle=":", linewidth=0.6, color="0.65", alpha=0.7)
    else:
        a0, a1 = axes
        X, Y, Z = evaluate_grid(name, dimension, axes, resolution)
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel(f"x{a0}")
        ax.set_ylabel(f"x{a1}")
        ax.set_zlabel("f(x)")
        ax.figure.colorbar(surf, ax=ax, shrink=0.55, aspect=10, pad=0.12)
    ax.set_title(title)


def build_combined_figure(names, dimension: int, axes, args, stem: str):
    if not names:
        return
    nrows, positions = grid_positions(len(names))
    fig = plt.figure(figsize=(11.0, 4.6 * nrows + 0.8))
    gs = fig.add_gridspec(nrows, 4)

    for name, (row, col0, col1) in zip(names, positions):
        projection = "3d" if len(axes) == 2 else None
        ax = fig.add_subplot(gs[row, col0:col1], projection=projection)
        draw_objective(ax, name, dimension, axes, args.resolution)

    fig.tight_layout()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}_dim{dimension}.{args.format}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="dimension N to instantiate each objective at (default: 2)",
    )
    p.add_argument(
        "--complex",
        action="store_true",
        help="also plot the 3 complex objectives (rastrigin, schwefel, michalewicz) "
        "in addition to the 5 simple ones (default: simple objectives only)",
    )
    p.add_argument(
        "--functions",
        nargs="+",
        default=None,
        help="override the objective-function list (default: simple set, "
        "plus complex set with --complex)",
    )
    p.add_argument(
        "--axes",
        type=int,
        nargs="+",
        default=None,
        help="which input axis/axes to sweep, 0-indexed (default: [0, 1], or "
        "[0] when --dimension 1); 1 axis makes a 2D line plot, 2 axes a 3D surface",
    )
    p.add_argument("--resolution", type=int, default=150, help="grid points per swept axis")
    p.add_argument(
        "--outdir", type=Path, default=ROOT / "report" / "objectives"
    )
    p.add_argument("--format", default="png", choices=["png", "pdf"])
    args = p.parse_args()

    if args.dimension < 1:
        sys.exit("--dimension must be >= 1")

    if args.axes is None:
        axes = [0] if args.dimension == 1 else [0, 1]
    else:
        axes = args.axes
    if len(axes) not in (1, 2):
        sys.exit("--axes must give exactly 1 or 2 indices")
    if any(a < 0 or a >= args.dimension for a in axes):
        sys.exit(f"--axes indices must be in [0, {args.dimension - 1}]")
    if len(axes) == 2 and args.dimension < 2:
        sys.exit("2 axes requires --dimension >= 2")

    style()

    if args.functions is not None:
        print(f"Plotting {len(args.functions)} objective(s) at dimension {args.dimension}: {args.functions}")
        build_combined_figure(args.functions, args.dimension, axes, args, "objectives")
        return

    print(f"Plotting {len(SIMPLE_FUNCTIONS)} simple objective(s) at dimension {args.dimension}: {SIMPLE_FUNCTIONS}")
    build_combined_figure(SIMPLE_FUNCTIONS, args.dimension, axes, args, "simple")

    if args.complex:
        print(f"Plotting {len(COMPLEX_FUNCTIONS)} complex objective(s) at dimension {args.dimension}: {COMPLEX_FUNCTIONS}")
        build_combined_figure(COMPLEX_FUNCTIONS, args.dimension, axes, args, "complex")


if __name__ == "__main__":
    main()
