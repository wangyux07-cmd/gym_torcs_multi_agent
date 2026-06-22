"""
pretrain_bc.py
==============
Behavior cloning pretraining: takes the demonstrations collected by
collect_demos.py and trains a PPO model's actor network to imitate them
via supervised learning (MSE loss), before any reinforcement learning
happens. The resulting model is saved and can be loaded by train.py as
a warm start, instead of starting PPO from random initialization.

How to run:
    1. python collect_demos.py   (produces demos.npz)
    2. python pretrain_bc.py     (produces ./checkpoints/bc_pretrained.zip)
    3. python train.py --warmstart   (starts PPO from this checkpoint)

Note: this script creates a PPO model object using the same environment
and hyperparameters as train.py, but instead of calling model.learn()
(reinforcement learning), it directly trains model.policy with a
supervised loss against the demonstration actions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

import config
from train import make_env

BC_EPOCHS = 20
BC_BATCH_SIZE = 256
BC_LEARNING_RATE = 1e-3


def main():
    print("Loading demonstrations from demos.npz...")
    data = np.load("demos.npz")
    states = torch.as_tensor(data["states"], dtype=torch.float32)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)
    print(f"Loaded {len(states)} demonstration pairs.")

    print("Creating environment and PPO model (untrained, random init)...")
    env = make_env()
    model = PPO("MlpPolicy", env, **config.PPO_PARAMS)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=BC_LEARNING_RATE)

    n_samples = len(states)
    n_batches = max(1, n_samples // BC_BATCH_SIZE)

    print(f"\nTraining actor network via behavior cloning for "
          f"{BC_EPOCHS} epochs...")
    for epoch in range(BC_EPOCHS):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * BC_BATCH_SIZE:(i + 1) * BC_BATCH_SIZE]
            batch_states = states[idx]
            batch_actions = actions[idx]

            # Get the policy's mean predicted action for these states.
            # SB3's ActorCriticPolicy exposes the pre-sampling distribution;
            # we use its mean as the "deterministic" prediction to match
            # against the demonstrated action.
            distribution = policy.get_distribution(batch_states)
            predicted_actions = distribution.distribution.mean

            loss = F.mse_loss(predicted_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch + 1:3d}/{BC_EPOCHS} -- MSE loss: {avg_loss:.5f}")

    save_path = "./checkpoints/bc_pretrained"
    model.save(save_path)
    print(f"\nBehavior-cloned model saved to {save_path}.zip")
    print("Run 'python train.py --warmstart' to start PPO from this checkpoint.")

    env.close()


if __name__ == "__main__":
    main()