"""
evaluate.py
===========
Run a trained PPO model in TORCS without any further learning.
Unlike train.py, this never calls model.learn() and never overwrites
any checkpoint -- it just drives the car with a fixed policy so you
can see how a given checkpoint performs. Per-episode CSV logging
(distRaced, terminal_reason, etc.) still happens automatically inside
TorcsEnv, same as during training.

IMPORTANT (fixed bug): you must pass --vecnormalize-stats pointing to
the stats file that matches the SAME step count as --model. Loading
mismatched stats (e.g. a model from step 840,000 paired with stats
saved at the very end of a much longer, possibly-collapsed run) feeds
the network observations normalized on a completely different scale
than it was trained on, and can make a genuinely good checkpoint
appear to perform terribly for reasons that have nothing to do with
the checkpoint itself. As of the updated train.py, VecNormalize stats
are now saved alongside each periodic model checkpoint with a matching
step count (e.g. "vecnormalize_840000_steps.pkl" next to
"torcs_ppo_840000_steps.zip") -- always pick the pair with matching
numbers. If you're evaluating a checkpoint from BEFORE this fix was
added to train.py, no exactly-matching stats file exists; results from
evaluating that checkpoint zero-shot are not reliable evidence of its
true quality -- prefer resuming training from it instead (VecNormalize
adapts online, so an initial mismatch matters far less there than in a
one-shot evaluation like this script).

How to run:
    1. Start TORCS, set up the race (Practice -> scr_server -> New Race).
    2. python evaluate.py --model ./checkpoints/torcs_ppo_840000_steps.zip \
           --vecnormalize-stats ./checkpoints/vecnormalize_840000_steps.pkl \
           --episodes 5
"""

import argparse
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from gym_torcs_env import TorcsEnv
from capped_policy import CappedActorCriticPolicy  # noqa: F401 -- imported so PPO.load() can deserialize checkpoints saved with this custom policy class


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent in TORCS")
    parser.add_argument("--model", default=config.TRAINING["model_save_path"] + ".zip",
                         help="Path to the .zip checkpoint to evaluate.")
    parser.add_argument("--vecnormalize-stats", default=None,
                         help="Path to the .pkl VecNormalize stats file matching --model's "
                              "step count. If omitted, falls back to config.NORMALIZE['stats_path'] "
                              "(the single end-of-training snapshot) -- only correct if --model IS "
                              "the final checkpoint from that same run.")
    parser.add_argument("--episodes", type=int, default=5,
                         help="Number of episodes to run.")
    parser.add_argument("--deterministic", action="store_true", default=True,
                         help="Use deterministic actions (default: on).")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    print("Creating environment...")
    print("Make sure TORCS is already running and showing the")
    print("blue 'waiting for connection' screen before continuing.\n")

    env = TorcsEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    if config.NORMALIZE["enabled"]:
        stats_path = args.vecnormalize_stats or config.NORMALIZE["stats_path"]
        if args.vecnormalize_stats is None:
            print("WARNING: no --vecnormalize-stats given, falling back to "
                  f"{stats_path} -- this is only correct if --model is the "
                  "final checkpoint of its training run. See the module "
                  "docstring for why a mismatch here silently produces "
                  "misleadingly bad evaluation results.\n")
        env = VecNormalize.load(stats_path, env)
        print(f"Loaded normalization stats from {stats_path}")
        # Freeze running stats -- evaluation should not keep updating them.
        env.training = False
        env.norm_reward = False

    print(f"Loading model from {args.model}")
    model = PPO.load(args.model, env=env)

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
        print(f"Episode {ep}/{args.episodes} finished, total_reward={total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()