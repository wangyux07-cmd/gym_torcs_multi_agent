from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from race_ai.config_io import load_controller_config
from race_ai.residual_env import ResidualDriverEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the rule controller through ResidualDriverEnv with fixed zero RL action."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--base-config", default="configs/rule_fast.json")
    parser.add_argument("--action-mode", choices=("speed_target", "slow_only", "speed_only", "full"), default="speed_target")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--damage-limit", type=float, default=5000.0)
    parser.add_argument("--stuck-steps-limit", type=int, default=300)
    parser.add_argument("--backward-angle-limit", type=float, default=1.25)
    parser.add_argument("--backward-speed-limit", type=float, default=-1.0)
    parser.add_argument("--max-speed-multiplier", type=float, default=1.08)
    parser.add_argument("--multiplier-step-limit", type=float, default=0.01)
    parser.add_argument("--caution-speed-multiplier", type=float, default=0.98)
    parser.add_argument("--boost-start-distance", type=float, default=650.0)
    parser.add_argument("--boost-deadband", type=float, default=0.25)
    parser.add_argument("--print-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # snakeoil parses sys.argv internally; keep our CLI flags away from it.
    sys.argv = [sys.argv[0]]

    config = load_controller_config(args.base_config)
    env = ResidualDriverEnv(
        host=args.host,
        port=args.port,
        config=config,
        target_laps=args.target_laps,
        max_steps=args.max_steps,
        damage_limit=args.damage_limit,
        stop_on_offtrack=True,
        stuck_steps_limit=args.stuck_steps_limit,
        backward_angle_limit=args.backward_angle_limit,
        backward_speed_limit=args.backward_speed_limit,
        action_mode=args.action_mode,
        max_speed_multiplier=args.max_speed_multiplier,
        multiplier_step_limit=args.multiplier_step_limit,
        caution_speed_multiplier=args.caution_speed_multiplier,
        boost_start_distance=args.boost_start_distance,
        boost_deadband=args.boost_deadband,
    )

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            total_reward = 0.0
            last_info = {}
            for step in range(args.max_steps):
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                last_info = info
                if args.print_every and step % args.print_every == 0:
                    print(
                        "episode={episode} step={step} reward={reward:.2f} total={total:.1f} "
                        "dist={dist:.1f} speed={speed:.1f} track_pos={track_pos:.3f} "
                        "angle={angle:.3f} mult={mult:.3f}".format(
                            episode=episode,
                            step=step,
                            reward=float(reward),
                            total=total_reward,
                            dist=float(info.get("dist_raced", 0.0)),
                            speed=float(info.get("speed_x", 0.0)),
                            track_pos=float(info.get("track_pos", 0.0)),
                            angle=float(info.get("angle", 0.0)),
                            mult=float(info.get("speed_multiplier", 1.0)),
                        )
                    )
                if terminated or truncated:
                    print(
                        "episode={episode} finished reason={reason} steps={steps} total_reward={reward:.1f} "
                        "dist={dist:.1f} best={best:.1f} lap={lap:.3f} track_pos={track_pos:.3f} "
                        "angle={angle:.3f} speed={speed:.1f} mean_mult={mean_mult:.4f} max_mult={max_mult:.4f}".format(
                            episode=episode,
                            reason=info.get("reason"),
                            steps=step + 1,
                            reward=total_reward,
                            dist=float(info.get("dist_raced", 0.0)),
                            best=float(info.get("best_progress", 0.0)),
                            lap=float(info.get("lap_time", 0.0)),
                            track_pos=float(info.get("track_pos", 0.0)),
                            angle=float(info.get("angle", 0.0)),
                            speed=float(info.get("speed_x", 0.0)),
                            mean_mult=float(info.get("mean_speed_multiplier", 1.0)),
                            max_mult=float(info.get("max_speed_multiplier", 1.0)),
                        )
                    )
                    break
            else:
                print(
                    "episode={episode} finished reason=max_steps steps={steps} total_reward={reward:.1f} "
                    "dist={dist:.1f} best={best:.1f}".format(
                        episode=episode,
                        steps=args.max_steps,
                        reward=total_reward,
                        dist=float(last_info.get("dist_raced", 0.0)),
                        best=float(last_info.get("best_progress", 0.0)),
                    )
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
