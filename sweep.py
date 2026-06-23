"""
sweep.py
========
Lightweight one-at-a-time parameter sweep tool.

Instead of manually editing config.py and re-running train.py for each
value you want to test (error-prone, easy to forget which value you
were testing), this script does it for you: it overrides ONE parameter,
runs a short training session, and logs the result into a summary file
so you end up with a clean comparison table -- which is also directly
usable as material for your thesis's ablation section.

This does NOT do a full grid search (testing every combination of every
parameter) -- it does one-at-a-time scanning, which is far cheaper and
is the appropriate choice given that each run takes real wall-clock time
on a single machine with a real-time simulator.

How to run:
    1. Start TORCS, set up the race (Practice -> scr_server -> New Race).
    2. python sweep.py --param w_anticipation --values 0.15,0.3,0.6 --steps 30000

    Repeat for each parameter you want to scan (restart TORCS each time).

What it does for each value:
    1. Overrides the named parameter in config (REWARD_WEIGHTS or
       REWARD_PARAMS, whichever dict contains it).
    2. Trains a FRESH PPO model (no warmstart/resume -- each value needs
       a clean, independent run to be a fair comparison) for the given
       number of steps.
    3. Reads back the final rollout stats and the terminal_reason
       distribution from this run's logs, and appends a row to
       sweep_results.csv.

IMPORTANT: each run uses its own checkpoint/log subfolder (named after
the parameter and value) so results don't overwrite each other.
"""

import argparse
import csv
import glob
import os
import sys
from collections import Counter

import config
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from train import make_env  # reuse the same env construction as train.py


def find_param_dict(param_name):
    """Figure out whether the parameter lives in REWARD_WEIGHTS or
    REWARD_PARAMS, since both are common targets for sweeps."""
    if param_name in config.REWARD_WEIGHTS:
        return config.REWARD_WEIGHTS
    if param_name in config.REWARD_PARAMS:
        return config.REWARD_PARAMS
    raise ValueError(
        f"'{param_name}' not found in REWARD_WEIGHTS or REWARD_PARAMS. "
        f"Check the spelling in config.py."
    )


def count_terminal_reasons(log_dir):
    """
    Scan every episode CSV in log_dir and count how each one ended
    (collision / out_of_track / stuck / wall_pinned / lap_complete / etc).
    This turns the raw per-step logs into the summary stat we actually
    care about for comparing sweep values.
    """
    counts = Counter()
    for filepath in glob.glob(os.path.join(log_dir, "*.csv")):
        with open(filepath, "r", newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            continue
        last_row = rows[-1]
        reason = last_row[-1].strip() if last_row else ""
        counts[reason if reason else "(empty)"] += 1
    return counts


def run_one_value(param_name, value, total_steps):
    """Train a fresh model with the given parameter override, return stats."""
    param_dict = find_param_dict(param_name)
    original_value = param_dict[param_name]
    param_dict[param_name] = value
    print(f"\n{'=' * 60}")
    print(f"  Sweep run: {param_name} = {value}")
    print(f"{'=' * 60}\n")

    # Give this run its own checkpoint/log folders so nothing overwrites.
    run_tag = f"{param_name}_{value}".replace(".", "p").replace("-", "neg")
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
        print("Interrupted -- using partial results for this value.")

    # Pull final rollout stats directly from the Monitor wrapper's buffer.
    ep_rewards = env.get_attr("episode_returns") if hasattr(env, "get_attr") else None
    ep_len_mean = None
    ep_rew_mean = None
    try:
        ep_info_buffer = model.ep_info_buffer
        if ep_info_buffer:
            ep_len_mean = sum(e["l"] for e in ep_info_buffer) / len(ep_info_buffer)
            ep_rew_mean = sum(e["r"] for e in ep_info_buffer) / len(ep_info_buffer)
    except Exception:
        pass

    reason_counts = count_terminal_reasons(config.LOGGING["log_dir"])

    env.close()
    param_dict[param_name] = original_value  # restore default for next run

    return {
        "param": param_name,
        "value": value,
        "ep_len_mean": ep_len_mean,
        "ep_rew_mean": ep_rew_mean,
        "reason_counts": dict(reason_counts),
    }


def append_result_row(result, csv_path="sweep_results.csv"):
    """
    Append one row to the running results CSV (creates header if new).

    IMPORTANT: uses a FIXED, canonical list of terminal_reason types so
    every row has the same columns in the same order, regardless of
    which reason types that particular run happened to produce. The
    earlier version derived column order from each run's own results,
    which silently misaligned columns whenever a later run encountered
    a reason type the first run hadn't (e.g. "(empty)" from a
    training_stopped marker) -- the header stayed fixed from run 1, but
    later rows had a different number/order of values, corrupting the
    table without any visible error.
    """
    KNOWN_REASONS = [
        "(empty)", "backward", "collision", "lap_complete",
        "out_of_track", "stuck", "wall_pinned",
    ]
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["param", "value", "ep_len_mean", "ep_rew_mean"] + KNOWN_REASONS)
        row = [
            result["param"],
            result["value"],
            result["ep_len_mean"],
            result["ep_rew_mean"],
        ] + [result["reason_counts"].get(r, 0) for r in KNOWN_REASONS]
        writer.writerow(row)

        # Warn if this run produced a reason type not in our known list,
        # so it doesn't silently get dropped from the row.
        unexpected = set(result["reason_counts"].keys()) - set(KNOWN_REASONS)
        if unexpected:
            print(f"  WARNING: unrecognized terminal_reason types seen: {unexpected} "
                  f"-- add them to KNOWN_REASONS in sweep.py to track them.")

    print(f"\nResult appended to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="One-at-a-time parameter sweep for TorcsEnv reward tuning")
    parser.add_argument("--param", required=True, help="Parameter name, e.g. w_anticipation")
    parser.add_argument("--values", required=True, help="Comma-separated values to test, e.g. 0.15,0.3,0.6")
    parser.add_argument("--steps", type=int, default=30000, help="Training steps per value (default 30000)")
    args, remaining = parser.parse_known_args()
    # IMPORTANT: snakeoil3_gym.py's Client class parses sys.argv directly
    # with its own getopt call (triggered whenever a Client is created,
    # i.e. every env.reset()). It doesn't know about --param/--values/
    # --steps, so without this cleanup it crashes the moment TorcsEnv
    # connects (same issue fixed earlier in train.py).
    sys.argv = [sys.argv[0]] + remaining

    values = [float(v) for v in args.values.split(",")]

    print(f"Sweeping '{args.param}' over values: {values}")
    print(f"Each run trains for {args.steps:,} steps.\n")
    print("Make sure TORCS is running and ready before each run starts.\n")

    for value in values:
        input(f"Press Enter when TORCS is ready for {args.param}={value}... ")
        result = run_one_value(args.param, value, args.steps)
        append_result_row(result)
        print(f"\n  {args.param}={value}: ep_len_mean={result['ep_len_mean']}, "
              f"ep_rew_mean={result['ep_rew_mean']}")
        print(f"  terminal reasons: {result['reason_counts']}")

    print(f"\nAll runs complete. See sweep_results.csv for the comparison table.")


if __name__ == "__main__":
    main()