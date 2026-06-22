"""
test_env.py
===========
Minimal smoke test for TorcsEnv. This script does NOT train anything.
Its only purpose is to confirm:
    1. The environment can connect to a running TORCS instance.
    2. reset() returns a valid 65-dim observation.
    3. step() with random actions runs without crashing.
    4. The episode CSV log file is created and written correctly.

How to run:
    1. Start TORCS manually.
    2. In the TORCS menu: Race -> Practice -> Configure Race
       -> pick any track -> set driver to scr_server -> Accept -> New Race
    3. TORCS should now show the blue "waiting for connection" screen.
    4. In a terminal, inside your project folder, run:
           python test_env.py
    5. Watch the TORCS window -- the car should start moving with
       random (probably bad/jerky) steering and throttle.
    6. Press Ctrl+C in the terminal at any time to stop early.
"""

import numpy as np
from gym_torcs_env import TorcsEnv


def main():
    print("=" * 60)
    print("TorcsEnv smoke test starting")
    print("Make sure TORCS is already running and showing the")
    print("blue 'waiting for connection' screen before this runs.")
    print("=" * 60)

    env = TorcsEnv()

    # ---- Step 1: reset() ----
    print("\n[1] Calling reset()...")
    obs, info = env.reset()
    print(f"    Observation shape: {obs.shape}  (expected: (65,))")
    print(f"    Observation dtype: {obs.dtype}")
    print(f"    First 5 values (angle, trackPos, speedX, speedY, speedZ):")
    print(f"    {obs[:5]}")
    assert obs.shape == (65,), "Observation shape is wrong!"
    print("    reset() OK")

    # ---- Step 2: step() loop with random actions ----
    # NOTE: increased to 400 steps (~8 seconds at 50Hz) to make sure the
    # connection stays alive through TORCS's countdown ("GO" moment).
    # Hypothesis being tested: the crash happens because the previous test
    # disconnected WHILE TORCS was still counting down, before the car was
    # actually released to drive.
    n_steps = 400
    print(f"\n[2] Running {n_steps} steps with random actions...")

    total_reward = 0.0
    for i in range(n_steps):
        action = env.action_space.sample()  # random steer/accel/brake
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 50 == 0:
            print(
                f"    step {i:4d} | reward={reward:7.3f} | "
                f"speedX={info['raw_obs'].get('speedX', 0):6.1f} | "
                f"trackPos={info['raw_obs'].get('trackPos', 0):6.3f} | "
                f"terminal_reason='{info['terminal_reason']}'",
                flush=True,
            )

        if terminated or truncated:
            print(f"    Episode ended at step {i} (reason: {info['terminal_reason']})", flush=True)
            print("    Calling reset() to start a new episode...", flush=True)
            obs, info = env.reset()

    print(f"\n[3] Test finished. Total reward over {n_steps} steps: {total_reward:.2f}")
    print("    (This number is meaningless right now -- actions were random.")
    print("     We only care that nothing crashed and values look sane.)")

    env.close()
    print("\nSmoke test complete. Check the logs/ folder for the CSV file(s).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Exiting.")
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        raise