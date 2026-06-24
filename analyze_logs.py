"""
analyze_logs.py
================
Reads all per-episode CSV logs and produces thesis-ready charts.

All metrics are computed from the existing CSV columns -- no new
logging is needed. The script is defensive about missing columns
(it will skip a chart rather than crash if a column doesn't exist).

How to run:
    python analyze_logs.py --log_dir ./logs --out_dir ./analysis_plots

Output (5 PNG files):
    1. reward_and_distance.png  -- per-episode total reward + distance raced
    2. collision_trend.png      -- terminal_reason counts over training
    3. speed_and_stability.png  -- average speed + trackPos deviation + steering jitter
    4. episode_length_dist.png  -- histogram of episode lengths
    5. trajectory_profile.png   -- trackPos vs distRaced for longest episodes

Data limitation (state this in your thesis):
    scr_server does not expose world (X,Y) coordinates. The trajectory
    plot uses trackPos vs distRaced as a commonly accepted proxy -- it
    shows lateral deviation along the track, not a true top-down view.
"""

import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_episode_number(filepath):
    match = re.search(r"ep(\d+)_", os.path.basename(filepath))
    return int(match.group(1)) if match else -1


def parse_timestamp(filepath):
    """
    Extract the YYYYMMDD_HHMMSS timestamp from the filename for TRUE
    chronological sorting. This matters because episode_count resets to
    1 every time training restarts -- if a log folder ever accumulates
    more than one training session (e.g. you forgot to rename the
    folder between runs), sorting by episode number alone interleaves
    different sessions' "episode 1", "episode 2", etc. on top of each
    other, producing misleading charts. Sorting by the real timestamp
    avoids this regardless of how many sessions are mixed in.
    """
    match = re.search(r"_(\d{8}_\d{6})\.csv$", os.path.basename(filepath))
    return match.group(1) if match else "00000000_000000"


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def load_episode(filepath):
    """Read one episode CSV, return a summary dict or None if unreadable."""
    try:
        with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None
    if not rows:
        return None

    rewards = []
    speeds = []
    track_positions = []
    dist_raced_vals = []
    steers = []

    for row in rows:
        rewards.append(safe_float(row.get("reward_total")))
        speeds.append(safe_float(row.get("speedX")))
        track_positions.append(safe_float(row.get("trackPos")))
        dist_raced_vals.append(safe_float(row.get("distRaced")))
        steers.append(safe_float(row.get("steer")))

    # Steering jitter: std of frame-to-frame steering changes
    steer_deltas = [abs(steers[i] - steers[i-1]) for i in range(1, len(steers))]

    last_row = rows[-1]
    terminal_reason = (last_row.get("terminal_reason") or "").strip()

    return {
        "filepath": filepath,
        "episode_num": parse_episode_number(filepath),
        "timestamp": parse_timestamp(filepath),
        "n_steps": len(rows),
        # Reward metrics
        "reward_total": sum(rewards),
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        # Speed metrics
        "speed_mean": float(np.mean(speeds)) if speeds else 0.0,
        "speed_max": float(np.max(speeds)) if speeds else 0.0,
        # Distance
        "dist_raced": dist_raced_vals[-1] if dist_raced_vals else 0.0,
        # Stability metrics
        "trackpos_deviation": float(np.mean([abs(tp) for tp in track_positions])) if track_positions else 0.0,
        "steer_jitter": float(np.std(steer_deltas)) if steer_deltas else 0.0,
        # Terminal
        "terminal_reason": terminal_reason if terminal_reason else "(empty)",
        # Raw series for trajectory plot
        "track_positions": track_positions,
        "dist_raced_series": dist_raced_vals,
    }


# ============================================================
# Chart 1: Reward + Distance over training
# ============================================================
def plot_reward_and_distance(episodes, out_dir):
    eps = sorted(episodes, key=lambda e: e["timestamp"])  # true chronological order
    if not eps:
        return
    x = list(range(len(eps)))  # chronological order index (robust to multi-session folders)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color1 = "#2196F3"
    ax1.set_xlabel("Episode (chronological order, not raw episode #)")
    ax1.set_ylabel("Total Reward", color=color1)
    ax1.plot(x, [e["reward_total"] for e in eps], color=color1, alpha=0.5, linewidth=0.8)
    # Rolling average for clarity
    window = min(20, len(eps) // 3) if len(eps) > 6 else 1
    if window > 1:
        rolling = np.convolve([e["reward_total"] for e in eps],
                              np.ones(window)/window, mode="valid")
        ax1.plot(x[window-1:], rolling, color=color1, linewidth=2, label=f"Reward (avg {window})")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#FF9800"
    ax2.set_ylabel("Distance Raced (m)", color=color2)
    ax2.plot(x, [e["dist_raced"] for e in eps], color=color2, alpha=0.5, linewidth=0.8)
    if window > 1:
        rolling_d = np.convolve([e["dist_raced"] for e in eps],
                                np.ones(window)/window, mode="valid")
        ax2.plot(x[window-1:], rolling_d, color=color2, linewidth=2, label=f"Distance (avg {window})")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Training Progress: Reward & Distance per Episode")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reward_and_distance.png"), dpi=150)
    plt.close()
    print("  Saved reward_and_distance.png")


# ============================================================
# Chart 2: Terminal reason trend (binned)
# ============================================================
def plot_collision_trend(episodes, out_dir, bin_size=20):
    eps = sorted(episodes, key=lambda e: e["timestamp"])  # true chronological order
    if not eps:
        return

    reasons = sorted(set(e["terminal_reason"] for e in eps))
    n_bins = max(1, len(eps) // bin_size)
    bin_edges = np.linspace(0, len(eps), n_bins + 1, dtype=int)

    fig, ax = plt.subplots(figsize=(12, 5))
    # Use stacked bar chart for clearer proportional view
    bottom = np.zeros(n_bins)
    color_map = {
        "collision": "#F44336",
        "out_of_track": "#FF9800",
        "stuck": "#9E9E9E",
        "wall_pinned": "#795548",
        "backward": "#9C27B0",
        "lap_complete": "#4CAF50",
        "(empty)": "#BDBDBD",
    }
    bar_x = range(n_bins)
    for reason in reasons:
        counts = []
        for i in range(n_bins):
            chunk = eps[bin_edges[i]:bin_edges[i + 1]]
            counts.append(sum(1 for e in chunk if e["terminal_reason"] == reason))
        counts = np.array(counts, dtype=float)
        color = color_map.get(reason, "#607D8B")
        ax.bar(bar_x, counts, bottom=bottom, label=reason, color=color, alpha=0.85)
        bottom += counts

    ax.set_xlabel(f"Chronological episode bin (~{bin_size} episodes each)")
    ax.set_ylabel("Count")
    ax.set_title("How Episodes Ended Over Training")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "collision_trend.png"), dpi=150)
    plt.close()
    print("  Saved collision_trend.png")


# ============================================================
# Chart 3: Speed, trackPos stability, steering jitter
# ============================================================
def plot_speed_and_stability(episodes, out_dir):
    eps = sorted(episodes, key=lambda e: e["timestamp"])  # true chronological order
    if not eps:
        return
    x = list(range(len(eps)))  # chronological order index (robust to multi-session folders)
    window = min(20, len(eps) // 3) if len(eps) > 6 else 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Average speed
    speeds = [e["speed_mean"] for e in eps]
    axes[0].plot(x, speeds, alpha=0.4, linewidth=0.8, color="#2196F3")
    if window > 1:
        rolling = np.convolve(speeds, np.ones(window)/window, mode="valid")
        axes[0].plot(x[window-1:], rolling, linewidth=2, color="#2196F3")
    axes[0].set_ylabel("Avg Speed (km/h)")
    axes[0].set_title("Speed Performance")
    axes[0].grid(alpha=0.3)

    # Panel 2: TrackPos deviation (lower = better centered)
    devs = [e["trackpos_deviation"] for e in eps]
    axes[1].plot(x, devs, alpha=0.4, linewidth=0.8, color="#FF9800")
    if window > 1:
        rolling = np.convolve(devs, np.ones(window)/window, mode="valid")
        axes[1].plot(x[window-1:], rolling, linewidth=2, color="#FF9800")
    axes[1].set_ylabel("Mean |trackPos|")
    axes[1].set_title("Track Centering (lower = closer to center)")
    axes[1].grid(alpha=0.3)

    # Panel 3: Steering jitter (lower = smoother)
    jitters = [e["steer_jitter"] for e in eps]
    axes[2].plot(x, jitters, alpha=0.4, linewidth=0.8, color="#4CAF50")
    if window > 1:
        rolling = np.convolve(jitters, np.ones(window)/window, mode="valid")
        axes[2].plot(x[window-1:], rolling, linewidth=2, color="#4CAF50")
    axes[2].set_ylabel("Steering Jitter (std of Δsteer)")
    axes[2].set_title("Driving Smoothness (lower = smoother)")
    axes[2].set_xlabel("Episode (chronological order, not raw episode #)")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Driving Quality Metrics Over Training", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speed_and_stability.png"), dpi=150)
    plt.close()
    print("  Saved speed_and_stability.png")


# ============================================================
# Chart 4: Episode length distribution
# ============================================================
def plot_episode_length_dist(episodes, out_dir):
    lengths = [e["n_steps"] for e in episodes]
    if not lengths:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=30, alpha=0.75, edgecolor="black", color="#2196F3")
    ax.axvline(np.mean(lengths), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(lengths):.0f}")
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Number of Episodes")
    ax.set_title("Episode Length Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "episode_length_dist.png"), dpi=150)
    plt.close()
    print("  Saved episode_length_dist.png")


# ============================================================
# Chart 5: Trajectory profile (trackPos vs distRaced)
# ============================================================
def plot_trajectory_profile(episodes, out_dir, n_samples=5):
    """
    NOTE FOR THESIS: this plots trackPos (lateral offset from center)
    against distRaced (distance along track), NOT true (X,Y) world
    coordinates -- scr_server doesn't provide those. This is a commonly
    accepted proxy showing how the car weaves relative to the track
    center, but it is not a literal overhead racing line.
    """
    eps_with_path = [e for e in episodes if len(e["track_positions"]) > 50]
    if not eps_with_path:
        print("  Skipping trajectory_profile.png -- not enough path data.")
        return

    # Pick the longest episodes (most likely to show interesting behavior)
    eps_with_path = sorted(eps_with_path, key=lambda e: e["n_steps"], reverse=True)[:n_samples]

    fig, ax = plt.subplots(figsize=(14, 5))
    for e in eps_with_path:
        label = f"ep {e['episode_num']} ({e['terminal_reason']}, {e['n_steps']} steps)"
        ax.plot(e["dist_raced_series"], e["track_positions"], alpha=0.7, linewidth=1, label=label)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="Centerline")
    ax.axhline(1, color="red", linestyle=":", linewidth=1, alpha=0.5, label="Track edge")
    ax.axhline(-1, color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Distance Raced (m)")
    ax.set_ylabel("trackPos (-1 to 1 = track edges)")
    ax.set_title("Lateral Deviation Profile (proxy for trajectory)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectory_profile.png"), dpi=150)
    plt.close()
    print("  Saved trajectory_profile.png")


# ============================================================
# Summary table (printed to console + saved as txt)
# ============================================================
def print_summary(episodes, out_dir):
    if not episodes:
        return

    reasons = {}
    for e in episodes:
        r = e["terminal_reason"]
        reasons[r] = reasons.get(r, 0) + 1

    summary = []
    summary.append(f"{'='*50}")
    summary.append(f"  TRAINING SUMMARY ({len(episodes)} episodes)")
    summary.append(f"{'='*50}")
    summary.append(f"  Avg episode length:   {np.mean([e['n_steps'] for e in episodes]):.0f} steps")
    summary.append(f"  Avg total reward:     {np.mean([e['reward_total'] for e in episodes]):.1f}")
    summary.append(f"  Avg reward std:       {np.mean([e['reward_std'] for e in episodes]):.2f}")
    summary.append(f"  Avg speed:            {np.mean([e['speed_mean'] for e in episodes]):.1f} km/h")
    summary.append(f"  Max speed seen:       {max(e['speed_max'] for e in episodes):.1f} km/h")
    summary.append(f"  Avg distance raced:   {np.mean([e['dist_raced'] for e in episodes]):.0f} m")
    summary.append(f"  Avg trackPos |dev|:   {np.mean([e['trackpos_deviation'] for e in episodes]):.3f}")
    summary.append(f"  Avg steering jitter:  {np.mean([e['steer_jitter'] for e in episodes]):.4f}")
    summary.append(f"  Terminal reasons:")
    for r in sorted(reasons):
        pct = reasons[r] / len(episodes) * 100
        summary.append(f"    {r:20s}: {reasons[r]:4d}  ({pct:.1f}%)")
    summary.append(f"{'='*50}")

    text = "\n".join(summary)
    print(text)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text)
    print(f"\n  Saved summary.txt")


def main():
    parser = argparse.ArgumentParser(description="Analyze TorcsEnv per-episode CSV logs")
    parser.add_argument("--log_dir", default="./logs", help="Folder containing episode CSV files")
    parser.add_argument("--out_dir", default="./analysis_plots", help="Where to save plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    filepaths = glob.glob(os.path.join(args.log_dir, "*.csv"))
    print(f"Found {len(filepaths)} CSV files in {args.log_dir}")
    if not filepaths:
        print("No CSV files found -- check --log_dir path.")
        return

    episodes = []
    for fp in sorted(filepaths):
        result = load_episode(fp)
        if result is not None:
            episodes.append(result)
    print(f"Successfully parsed {len(episodes)} episodes.\n")

    print("Generating charts...")
    plot_reward_and_distance(episodes, args.out_dir)
    plot_collision_trend(episodes, args.out_dir)
    plot_speed_and_stability(episodes, args.out_dir)
    plot_episode_length_dist(episodes, args.out_dir)
    plot_trajectory_profile(episodes, args.out_dir)

    print("\n")
    print_summary(episodes, args.out_dir)

    print(f"\nDone. All outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()