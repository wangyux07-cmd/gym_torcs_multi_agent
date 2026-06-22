"""
train.py
========
Single-car PPO training script. This is what you run to actually
train the agent -- it is separate from test_env.py (which only
checks that the connection works, without learning anything).

What this script does:
    1. Wraps TorcsEnv so stable-baselines3 can use it.
    2. Creates a PPO agent (using hyperparameters from config.py).
    3. Trains it, automatically saving a checkpoint every N steps.
    4. If training is interrupted (Ctrl+C, crash, power loss), you can
       resume from the latest checkpoint instead of starting over.

How to run a fresh training run:
    1. Start TORCS, set up the race (Practice -> scr_server -> New Race).
    2. python train.py

How to resume an interrupted run:
    python train.py --resume

Monitoring training progress (optional, in a separate terminal):
    tensorboard --logdir ./tb_logs
    Then open http://localhost:6006 in your browser.
"""

import argparse
import glob
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from gym_torcs_env import TorcsEnv


def make_env():
    """
    Create and wrap the environment.

    Monitor records per-episode reward/length statistics, which is what
    lets stable-baselines3 (and TensorBoard) show meaningful training
    curves -- without it, you would only see raw per-step rewards.

    VecNormalize tracks a running mean/std of observations and rewards
    and rescales them on the fly. This was added after several training
    runs showed value_loss climbing steadily while explained_variance
    stayed near zero -- a sign the Critic network couldn't find a stable
    scale to predict against, since raw speeds/rewards range from single
    digits to the hundreds depending on context.
    """
    env = TorcsEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    if config.NORMALIZE["enabled"]:
        env = VecNormalize(
            env,
            norm_obs=config.NORMALIZE["norm_obs"],
            norm_reward=config.NORMALIZE["norm_reward"],
            clip_obs=config.NORMALIZE["clip_obs"],
            clip_reward=config.NORMALIZE["clip_reward"],
        )
    return env


def find_latest_checkpoint(checkpoint_dir):
    """
    Look in the checkpoint directory for the most recently saved model
    and return its path, or None if no checkpoints exist yet.
    """
    pattern = os.path.join(checkpoint_dir, "*.zip")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # Sort by modification time, newest last.
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent to drive in TORCS")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint instead of starting fresh.",
    )
    parser.add_argument(
        "--warmstart",
        action="store_true",
        help="Start from the behavior-cloned model (./checkpoints/bc_pretrained.zip) "
             "instead of random initialization. Run pretrain_bc.py first to create it.",
    )
    args, remaining = parser.parse_known_args()
    # IMPORTANT: snakeoil3_gym.py's Client class parses sys.argv directly
    # with its own getopt call (triggered whenever a Client is created,
    # i.e. every env.reset()). It doesn't know about --resume/--warmstart,
    # so without this cleanup it crashes the moment TorcsEnv connects.
    sys.argv = [sys.argv[0]] + remaining

    # --- Prepare directories ---
    os.makedirs(config.TRAINING["checkpoint_dir"], exist_ok=True)
    os.makedirs(config.TRAINING["tensorboard_log_dir"], exist_ok=True)

    # --- Build the environment ---
    print("Creating environment...")
    print("Make sure TORCS is already running and showing the")
    print("blue 'waiting for connection' screen before continuing.\n")
    env = make_env()

    # --- Create or load the PPO model ---
    if args.resume:
        latest = find_latest_checkpoint(config.TRAINING["checkpoint_dir"])
        if latest is None:
            print("No checkpoint found -- starting a fresh training run instead.")
            model = PPO(
                "MlpPolicy",
                env,
                tensorboard_log=config.TRAINING["tensorboard_log_dir"],
                **config.PPO_PARAMS,
            )
        else:
            print(f"Resuming from checkpoint: {latest}")
            model = PPO.load(latest, env=env)
            # Restore VecNormalize running statistics if they were saved.
            stats_path = config.NORMALIZE["stats_path"]
            if config.NORMALIZE["enabled"] and os.path.exists(stats_path):
                env = VecNormalize.load(stats_path, env.venv)
                model.set_env(env)
                print(f"Restored normalization statistics from {stats_path}")
    else:
        if args.warmstart:
            bc_path = "./checkpoints/bc_pretrained.zip"
            print(f"Starting from behavior-cloned weights: {bc_path}")
            print("(Run pretrain_bc.py first if this file doesn't exist yet.)")
            model = PPO.load(bc_path, env=env)
        else:
            print("Starting a fresh training run.")
            model = PPO(
                "MlpPolicy",
                env,
                tensorboard_log=config.TRAINING["tensorboard_log_dir"],
                **config.PPO_PARAMS,
            )

    # --- Checkpoint callback: saves the model periodically ---
    checkpoint_callback = CheckpointCallback(
        save_freq=config.TRAINING["checkpoint_freq"],
        save_path=config.TRAINING["checkpoint_dir"],
        name_prefix="torcs_ppo",
    )

    # --- Train ---
    print(f"\nTraining for {config.TRAINING['total_timesteps']:,} timesteps...")
    print(f"Checkpoints will be saved every {config.TRAINING['checkpoint_freq']:,} "
          f"steps to {config.TRAINING['checkpoint_dir']}\n")
    print("Press Ctrl+C at any time to stop -- your latest checkpoint will")
    print("already be saved, so you can resume later with --resume.\n")

    try:
        model.learn(
            total_timesteps=config.TRAINING["total_timesteps"],
            callback=checkpoint_callback,
            reset_num_timesteps=not args.resume,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model state...")

    # --- Always save a final snapshot, even if interrupted ---
    model.save(config.TRAINING["model_save_path"])
    print(f"\nModel saved to {config.TRAINING['model_save_path']}.zip")

    if config.NORMALIZE["enabled"]:
        env.save(config.NORMALIZE["stats_path"])
        print(f"Normalization stats saved to {config.NORMALIZE['stats_path']}")

    env.close()


if __name__ == "__main__":
    main()