"""
check_reward_composition.py
=============================
Reads episode CSV logs and computes, for each episode, the ACTUAL
weighted contribution of every reward component to reward_total --
not the raw per-step values logged in the CSV (those are pre-weight),
but raw_sum * weight, matching how _compute_reward actually assembles
the total.

This directly answers: is speed reward dominating the total, with
checkpoint/milestone/record rewards too small to matter? Rather than
guessing from theory, this measures it from real logged episodes.

How to run (from the project directory, so it can import config.py):
    python check_reward_composition.py --log_dir ./logs --n_episodes 20
"""

import argparse
import csv
import glob
import os

import numpy as np

import config


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def load_rows(filepath):
    try:
        with open(filepath, "r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Break down reward_total into each weighted component, per episode")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--n_episodes", type=int, default=20,
                         help="How many of the MOST RECENT episodes to analyze (by file mtime)")
    args = parser.parse_args()

    w = config.REWARD_WEIGHTS

    filepaths = glob.glob(os.path.join(args.log_dir, "*.csv"))
    filepaths.sort(key=os.path.getmtime)
    filepaths = filepaths[-args.n_episodes:] if len(filepaths) > args.n_episodes else filepaths

    print(f"Analyzing the {len(filepaths)} most recent episodes in {args.log_dir}\n")

    # Component name -> (csv column, weight key or None if unweighted)
    components = {
        "speed":        ("r_speed",        "w_speed"),
        "safety":       ("r_safety",       "w_safety"),
        "smooth":       ("r_smooth",       "w_smooth"),
        "anticipation": ("r_anticipation", "w_anticipation"),
        "progress":     ("r_progress",     "w_progress"),
        "time":         ("r_time",         None),  # added unweighted
        "terminal":     ("r_terminal",     None),  # added unweighted
        "checkpoint":   ("r_checkpoint",   None),  # added unweighted
        "record":       ("r_record",       None),  # added unweighted
    }

    all_episode_totals = {name: [] for name in components}
    all_episode_totals["reward_total_logged"] = []
    all_episode_totals["distRaced_final"] = []

    for fp in filepaths:
        rows = load_rows(fp)
        if not rows:
            continue

        sums = {name: 0.0 for name in components}
        for row in rows:
            for name, (col, weight_key) in components.items():
                raw_val = safe_float(row.get(col))
                weight = w[weight_key] if weight_key else 1.0
                sums[name] += raw_val * weight

        for name in components:
            all_episode_totals[name].append(sums[name])
        all_episode_totals["reward_total_logged"].append(
            sum(safe_float(row.get("reward_total")) for row in rows)
        )
        all_episode_totals["distRaced_final"].append(safe_float(rows[-1].get("distRaced")) if rows else 0.0)

    n = len(filepaths)
    if n == 0:
        print("No episodes found -- check --log_dir.")
        return

    # Reconstructed total = sum of all weighted components (should be
    # close to, but not necessarily exactly equal to, the logged
    # reward_total, since reward_total is the actual cumulative reward
    # SB3 saw including any floating-point step-by-step accumulation --
    # this is a sanity check, not expected to match to the decimal).
    reconstructed_total = sum(np.mean(all_episode_totals[name]) for name in components)

    print("=" * 70)
    print(f"  AVERAGE REWARD COMPOSITION across these {n} episodes")
    print("=" * 70)
    print(f"  Avg final distRaced: {np.mean(all_episode_totals['distRaced_final']):.1f} m\n")

    for name in components:
        avg = np.mean(all_episode_totals[name])
        pct = (avg / reconstructed_total * 100) if reconstructed_total != 0 else 0.0
        print(f"  {name:>12}: {avg:>12,.1f}   ({pct:>5.1f}% of reconstructed total)")

    print(f"  {'-'*12}   {'-'*12}")
    print(f"  {'RECONSTRUCTED':>12}: {reconstructed_total:>12,.1f}")
    print(f"  {'LOGGED (SB3)':>12}: {np.mean(all_episode_totals['reward_total_logged']):>12,.1f}  "
          f"(should be close to reconstructed -- large gap = a bug/missing term somewhere)")

    print("\nHow to read this:")
    print("  - If 'speed' is e.g. 70%+ of the total and 'checkpoint'/'record'")
    print("    are each under 5%, the milestone/record rewards are likely too")
    print("    small to meaningfully compete with speed -- raise their weight")
    print("    or magnitude, OR lower w_speed further.")
    print("  - 'checkpoint' and 'record' being small is somewhat EXPECTED by")
    print("    design (they're meant to be occasional bonuses, not the main")
    print("    driver) -- the real question is whether they're large enough")
    print("    to be a noticeable nudge AT THE MOMENT they fire, not whether")
    print("    they dominate the episode average (they're not supposed to).")


if __name__ == "__main__":
    main()