"""
collect_demos.py
=================
Runs the existing rule-based driver (the same logic as drive_modular() in
torcs_jm_par.py) through our TorcsEnv to collect (state, action) pairs.
These demonstrations are used by pretrain_bc.py to give the PPO policy a
behavior-cloned head start before reinforcement learning begins.

IMPORTANT CAVEAT: TorcsEnv's observation is already normalized to roughly
[-1, 1] (see _build_state_vector). During actual PPO training, VecNormalize
applies an additional layer of normalization on top of that. This means the
pretrained network's input scale won't exactly match what PPO sees during
RL -- the warm-started weights are a reasonable head start, not a perfect
match. PPO will need some further training to fully adapt, which is normal
and expected.

How to run:
    1. Start TORCS, set up the race (Practice -> scr_server -> New Race).
    2. python collect_demos.py
    Output: demos.npz (state/action pairs) in the current directory.
"""

import math
import numpy as np

from gym_torcs_env import TorcsEnv

# ===================== Rule-based controller (ported from torcs_jm_par.py) =====================
# Same constants as the original drive_modular() in torcs_jm_par.py.
TARGET_SPEED = 120
STEER_GAIN = 30
CENTERING_GAIN = 0.20
BRAKE_THRESHOLD = 0.9


def rule_based_action(raw_obs, prev_accel):
    """
    Reimplements the logic of drive_modular() / calculate_steering() /
    calculate_throttle() / apply_brakes() from torcs_jm_par.py, but reads
    from the raw_obs dict (as exposed in TorcsEnv's info["raw_obs"])
    instead of the snakeoil3 Client object directly.

    Returns: np.array([steer, accel, brake], dtype=np.float32)
    """
    angle = raw_obs.get("angle", 0.0)
    track_pos = raw_obs.get("trackPos", 0.0)
    speed_x = raw_obs.get("speedX", 0.0)

    steer = (angle * STEER_GAIN / math.pi) - (track_pos * CENTERING_GAIN)
    steer = max(-1.0, min(1.0, steer))

    if speed_x < TARGET_SPEED - (steer * 2.5):
        accel = min(1.0, prev_accel + 0.4)
    else:
        accel = max(0.0, prev_accel - 0.2)
    if speed_x < 10:
        accel += 1 / (speed_x + 0.1)
    accel = max(0.0, min(1.0, accel))

    brake = 0.3 if abs(angle) > BRAKE_THRESHOLD else 0.0

    return np.array([steer, accel, brake], dtype=np.float32), accel


def main(num_episodes=20, max_steps_per_episode=3000):
    env = TorcsEnv()

    all_states = []
    all_actions = []

    for ep in range(num_episodes):
        state, info = env.reset()
        prev_accel = 0.0
        print(f"Episode {ep + 1}/{num_episodes} -- collecting demonstrations...")

        for step in range(max_steps_per_episode):
            raw_obs = info["raw_obs"]
            action, prev_accel = rule_based_action(raw_obs, prev_accel)

            all_states.append(state.copy())
            all_actions.append(action.copy())

            state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"  Episode ended after {step + 1} steps "
                      f"(reason: {info['terminal_reason']})")
                break

    env.close()

    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)

    print(f"\nCollected {len(states)} (state, action) pairs "
          f"from {num_episodes} episodes.")
    np.savez("demos.npz", states=states, actions=actions)
    print("Saved to demos.npz")


if __name__ == "__main__":
    main()