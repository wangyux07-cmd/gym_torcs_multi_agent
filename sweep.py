"""
sweep.py
========
Parameter sweep tool supporting both one-at-a-time and 2D grid search.

One-at-a-time mode (single --param):
    python sweep.py --param w_anticipation --values 0,0.15,0.3 --steps 30000

Grid search mode (two --param flags):
    python sweep.py --param w_anticipation --values 0,0.15,0.3 \
                    --param2 w_safety --values2 0.5,1.0,1.5,2.0 \
                    --steps 30000

Grid mode tests ALL combinations of the two parameters (e.g. 3x4=12 runs),
producing a table that shows how the two parameters interact -- something
one-at-a-time sweeps can't reveal. The output CSV is directly usable as
a thesis grid-search table and as source data for a Pareto frontier plot.

Each run:
    1. Overrides the named parameter(s) in config.
    2. Trains a FRESH PPO model for the given number of steps.
    3. Reads the episode logs to compute avg speed, collision count, etc.
    4. Appends a row to sweep_results.csv.
"""

import argparse
import csv
import glob
import os
import sys
from collections import Counter

import numpy as np

import config
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from train import make_env


def find_param_dict(param_name):
    """Figure out whether the parameter lives in REWARD_WEIGHTS, REWARD_PARAMS, or PPO_PARAMS."""
    if param_name in config.REWARD_WEIGHTS:
        return config.REWARD_WEIGHTS
    if param_name in config.REWARD_PARAMS:
        return config.REWARD_PARAMS
    if param_name in config.PPO_PARAMS:
        return config.PPO_PARAMS
    raise ValueError(
        f"'{param_name}' not found in REWARD_WEIGHTS, REWARD_PARAMS, or PPO_PARAMS."
    )


def count_terminal_reasons(log_dir):
    """Scan every episode CSV in log_dir and count terminal reasons."""
    counts = Counter()
    for filepath in glob.glob(os.path.join(log_dir, "*.csv")):
        with open(filepath, "r", newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            continue
        last_row = rows[-1]
        reason = last_row[-1].strip() if last_row and last_row[-1] else ""
        counts[reason if reason else "(empty)"] += 1
    return counts


def compute_avg_speed(log_dir):
    """
    Read all episode CSVs in log_dir and compute overall average speedX.
    This is needed for the Pareto frontier plot (speed vs safety tradeoff).
    """
    all_speeds = []
    for filepath in glob.glob(os.path.join(log_dir, "*.csv")):
        try:
            with open(filepath, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        all_speeds.append(float(row.get("speedX", 0)))
                    except (ValueError, TypeError):
                        pass
        except Exception:
            continue
    return float(np.mean(all_speeds)) if all_speeds else 0.0


def run_one_combination(params_dict, total_steps):
    """
    Train a fresh model with the given parameter overrides.

    params_dict: e.g. {"w_anticipation": 0.3, "w_safety": 1.0}
    Returns a result dict with metrics.
    """
    # Apply overrides and remember originals for restoration
    originals = {}
    for param_name, value in params_dict.items():
        d = find_param_dict(param_name)
        originals[(param_name, id(d))] = (d, param_name, d[param_name])
        d[param_name] = value

    # Build a descriptive tag for this run, automatically prefixed with
    # config.EXPERIMENT_TAG so every folder name carries "which version
    # of the reward logic this was" without you needing to invent a name.
    tag_parts = [f"{k}_{v}".replace(".", "p").replace("-", "neg")
                 for k, v in params_dict.items()]
    run_tag = f"{config.EXPERIMENT_TAG}__" + "__".join(tag_parts)

    print(f"\n{'=' * 60}")
    print(f"  Sweep run: {params_dict}")
    print(f"{'=' * 60}\n")

    # Separate checkpoint/log folders per run
    config.TRAINING["checkpoint_dir"] = f"./sweep_checkpoints/{run_tag}"
    config.LOGGING["log_dir"] = f"./sweep_logs/{run_tag}"
    os.makedirs(config.TRAINING["checkpoint_dir"], exist_ok=True)
    os.makedirs(config.LOGGING["log_dir"], exist_ok=True)

    env = make_env()
    model = PPO("MlpPolicy", env, **config.PPO_PARAMS)
    checkpoint_callback = CheckpointCallback(
        save_freq=config.TRAINING["checkpoint_freq"],
        save_path=config.TRAINING["checkpoint_dir"],
        name_prefix="ppo",
    )

    try:
        model.learn(total_timesteps=total_steps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Interrupted -- using partial results.")

    # Collect metrics
    ep_len_mean = None
    ep_rew_mean = None
    try:
        buf = model.ep_info_buffer
        if buf:
            ep_len_mean = sum(e["l"] for e in buf) / len(buf)
            ep_rew_mean = sum(e["r"] for e in buf) / len(buf)
    except Exception:
        pass

    reason_counts = count_terminal_reasons(config.LOGGING["log_dir"])
    avg_speed = compute_avg_speed(config.LOGGING["log_dir"])

    env.close()

    # Restore originals
    for (_, _), (d, name, orig_val) in originals.items():
        d[name] = orig_val

    return {
        "params": params_dict,
        "ep_len_mean": ep_len_mean,
        "ep_rew_mean": ep_rew_mean,
        "avg_speed": avg_speed,
        "reason_counts": dict(reason_counts),
    }


KNOWN_REASONS = [
    "(empty)", "backward", "collision", "lap_complete",
    "out_of_track", "stuck", "wall_pinned",
]


def append_result_row(result, param_names, csv_path="sweep_results.csv"):
    """
    Append one row to the results CSV. Header adapts to whether this is
    a 1-param sweep or a 2-param grid search.
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = list(param_names) + [
                "ep_len_mean", "ep_rew_mean", "avg_speed"
            ] + KNOWN_REASONS
            writer.writerow(header)

        row = [result["params"][p] for p in param_names] + [
            result["ep_len_mean"],
            result["ep_rew_mean"],
            result["avg_speed"],
        ] + [result["reason_counts"].get(r, 0) for r in KNOWN_REASONS]
        writer.writerow(row)

        unexpected = set(result["reason_counts"].keys()) - set(KNOWN_REASONS)
        if unexpected:
            print(f"  WARNING: unrecognized terminal_reason types: {unexpected}")

    print(f"\nResult appended to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep (one-at-a-time or 2D grid search)"
    )
    parser.add_argument("--param", required=True,
                        help="First parameter name, e.g. w_anticipation")
    parser.add_argument("--values", required=True,
                        help="Comma-separated values for first param")
    parser.add_argument("--param2", default=None,
                        help="Second parameter name for grid search (optional)")
    parser.add_argument("--values2", default=None,
                        help="Comma-separated values for second param")
    parser.add_argument("--steps", type=int, default=30000,
                        help="Training steps per combination (default 30000)")

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    values1 = [float(v) for v in args.values.split(",")]

    # Build list of all (param_dict) combinations to test
    if args.param2 and args.values2:
        # Grid search mode
        values2 = [float(v) for v in args.values2.split(",")]
        combinations = []
        for v1 in values1:
            for v2 in values2:
                combinations.append({args.param: v1, args.param2: v2})
        param_names = [args.param, args.param2]
        print(f"GRID SEARCH: {args.param} x {args.param2}")
        print(f"  {args.param}:  {values1}")
        print(f"  {args.param2}: {values2}")
        print(f"  Total combinations: {len(combinations)}")
    else:
        # One-at-a-time mode
        combinations = [{args.param: v} for v in values1]
        param_names = [args.param]
        print(f"ONE-AT-A-TIME: {args.param} over {values1}")

    print(f"  Steps per run: {args.steps:,}")
    print(f"  Estimated total time: ~{len(combinations) * args.steps / 36 / 60:.0f} minutes")
    print(f"\nMake sure TORCS is running and ready before each run.\n")

    csv_path = f"sweep_results_{config.EXPERIMENT_TAG}.csv"
    print(f"Results will be saved to: {csv_path}")
    print(f"(Tag comes from config.EXPERIMENT_TAG -- update that when the "
          f"reward formula/track changes, so old and new results never mix.)\n")

    for i, combo in enumerate(combinations):
        label = ", ".join(f"{k}={v}" for k, v in combo.items())
        input(f"[{i+1}/{len(combinations)}] Press Enter when TORCS is ready for {label}... ")

        result = run_one_combination(combo, args.steps)
        append_result_row(result, param_names, csv_path=csv_path)

        print(f"\n  {label}:")
        print(f"    ep_len_mean = {result['ep_len_mean']}")
        print(f"    ep_rew_mean = {result['ep_rew_mean']}")
        print(f"    avg_speed   = {result['avg_speed']:.1f} km/h")
        print(f"    terminal reasons: {result['reason_counts']}")

    print(f"\nAll {len(combinations)} runs complete.")
    print(f"Results saved to {csv_path}")
    print(f"Don't forget to add a row to EXPERIMENT_LOG.md describing this batch.")


if __name__ == "__main__":
    main()