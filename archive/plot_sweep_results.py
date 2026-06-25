"""
plot_sweep_results.py
======================
Plots for sweep_results.csv (the GRID SEARCH summary table -- one row
per parameter combination), as opposed to analyze_logs.py which plots
per-episode data from a single training run's logs. These are
different data shapes and need different plotting logic.

How to run:
    python plot_sweep_results.py --csv sweep_results.csv --out_dir ./sweep_plots

Output:
    1. heatmap_ep_rew_mean.png   -- grid heatmap of reward across (param1, param2)
    2. heatmap_collisions.png    -- grid heatmap of collision count
    3. pareto_frontier.png       -- avg_speed vs collision count scatter,
                                     with the actual Pareto-optimal points
                                     highlighted and connected
    4. ablation_bar.png          -- bar chart comparing param1=0 (ablated)
                                     vs param1>0 rows, averaged over param2
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def detect_param_columns(rows):
    """
    The first two columns of sweep_results.csv are always the swept
    parameter names (see sweep.py's append_result_row). Everything
    else (ep_len_mean, ep_rew_mean, avg_speed, and the reason columns)
    comes after.
    """
    fieldnames = list(rows[0].keys())
    known_metric_cols = {"ep_len_mean", "ep_rew_mean", "avg_speed",
                          "(empty)", "backward", "collision", "lap_complete",
                          "out_of_track", "stuck", "wall_pinned"}
    param_cols = [c for c in fieldnames if c not in known_metric_cols]
    return param_cols


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def plot_heatmap(rows, param1, param2, metric, out_path, title, cmap="viridis"):
    p1_values = sorted(set(safe_float(r[param1]) for r in rows))
    p2_values = sorted(set(safe_float(r[param2]) for r in rows))

    grid = np.full((len(p1_values), len(p2_values)), np.nan)
    for r in rows:
        i = p1_values.index(safe_float(r[param1]))
        j = p2_values.index(safe_float(r[param2]))
        grid[i, j] = safe_float(r.get(metric))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(p2_values)))
    ax.set_xticklabels(p2_values)
    ax.set_yticks(range(len(p1_values)))
    ax.set_yticklabels(p1_values)
    ax.set_xlabel(param2)
    ax.set_ylabel(param1)
    ax.set_title(title)

    # Annotate each cell with its value
    for i in range(len(p1_values)):
        for j in range(len(p2_values)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                         color="white", fontsize=9,
                         bbox=dict(facecolor="black", alpha=0.3, pad=1))

    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_pareto_frontier(rows, out_path):
    """
    Speed vs Safety tradeoff: x = avg_speed (higher is better),
    y = collision count (lower is better). A point is Pareto-optimal
    if no other point has BOTH higher speed AND fewer collisions.
    """
    points = [(safe_float(r.get("avg_speed")), safe_float(r.get("collision")), r)
              for r in rows]

    def is_pareto_optimal(p, all_points):
        speed, coll, _ = p
        for s2, c2, _ in all_points:
            if s2 >= speed and c2 <= coll and (s2 > speed or c2 < coll):
                return False
        return True

    pareto_points = [p for p in points if is_pareto_optimal(p, points)]
    pareto_points.sort(key=lambda p: p[0])

    fig, ax = plt.subplots(figsize=(9, 6))
    all_x = [p[0] for p in points]
    all_y = [p[1] for p in points]
    ax.scatter(all_x, all_y, alpha=0.5, s=60, color="gray", label="All combinations")

    px = [p[0] for p in pareto_points]
    py = [p[1] for p in pareto_points]
    ax.plot(px, py, color="red", marker="o", linewidth=2, markersize=10,
             label="Pareto frontier", zorder=5)

    # Label each Pareto point with its parameter combination
    param_cols = detect_param_columns(rows)
    for speed, coll, r in pareto_points:
        label = ", ".join(f"{c}={r[c]}" for c in param_cols)
        ax.annotate(label, (speed, coll), textcoords="offset points",
                    xytext=(8, 5), fontsize=7)

    ax.set_xlabel("Average Speed (km/h) -- higher is better")
    ax.set_ylabel("Collision Count -- lower is better")
    ax.set_title("Speed vs. Safety Pareto Frontier")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_ablation_bar(rows, param1, out_path):
    """
    Compares param1=0 (mechanism ablated/removed) against param1>0
    (mechanism present), averaging over all other parameter values.
    This directly answers "is this mechanism useful?"
    """
    ablated = [r for r in rows if safe_float(r[param1]) == 0.0]
    present = [r for r in rows if safe_float(r[param1]) != 0.0]

    if not ablated or not present:
        print(f"  Skipping ablation_bar.png -- need both {param1}=0 and {param1}>0 rows.")
        return

    metrics = ["ep_rew_mean", "avg_speed", "collision", "stuck", "out_of_track"]
    ablated_means = [np.mean([safe_float(r.get(m)) for r in ablated]) for m in metrics]
    present_means = [np.mean([safe_float(r.get(m)) for r in present]) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, ablated_means, width, label=f"{param1}=0 (ablated)", color="#F44336", alpha=0.8)
    ax.bar(x + width/2, present_means, width, label=f"{param1}>0 (present)", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_title(f"Ablation: Effect of Removing {param1}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot grid search results from sweep_results.csv")
    parser.add_argument("--csv", default="sweep_results.csv")
    parser.add_argument("--out_dir", default="./sweep_plots")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_results(args.csv)
    param_cols = detect_param_columns(rows)
    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Detected parameter columns: {param_cols}")

    if len(param_cols) < 2:
        print("Only one parameter column found -- heatmaps need two parameters "
              "(this looks like one-at-a-time sweep data, not a grid search).")
        return

    param1, param2 = param_cols[0], param_cols[1]

    print("\nGenerating plots...")
    plot_heatmap(rows, param1, param2, "ep_rew_mean", f"{args.out_dir}/heatmap_ep_rew_mean.png",
                 f"Mean Episode Reward: {param1} x {param2}")
    plot_heatmap(rows, param1, param2, "collision", f"{args.out_dir}/heatmap_collisions.png",
                 f"Collision Count: {param1} x {param2}", cmap="Reds")
    plot_pareto_frontier(rows, f"{args.out_dir}/pareto_frontier.png")
    plot_ablation_bar(rows, param1, f"{args.out_dir}/ablation_bar.png")

    print(f"\nDone. All plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()