from __future__ import annotations

import argparse
import sys
from pathlib import Path

from race_ai.config_io import load_controller_config
from race_ai.sector_reference import load_sector_reference
from race_ai.sector_residual_env import SectorResidualEnv
from race_ai.speed_profile import load_speed_profile


def load_sac(path: str):
    try:
        from stable_baselines3 import SAC
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3 is required to run a sector SAC model. Install it with:\n"
            "  python -m pip install stable-baselines3 gymnasium torch\n"
            f"Original error: {exc}"
        ) from exc
    return SAC.load(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained sector-level SAC residual policy in TORCS.")
    parser.add_argument("--model", default="models/sector_sac_speed_only/final_model.zip")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--base-config", default="configs/rule_fast.json")
    parser.add_argument("--base-profile", default="configs/speed_profile_v2_12_finish_brake_170.json")
    parser.add_argument("--reference", default="configs/sector_reference_v2_12_50m.json")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument("--speed-delta", type=float, default=0.04)
    parser.add_argument("--brake-delta-kmh", type=float, default=5.0)
    parser.add_argument("--enable-brake-delta", action="store_true")
    parser.add_argument("--locked-sectors", default="44-50,61-66")
    parser.add_argument("--max-speed-multiplier", type=float, default=1.44)
    parser.add_argument("--max-brake-target-speed", type=float, default=170.0)
    parser.add_argument("--offtrack-speed-penalty-scale", type=float, default=0.9)
    parser.add_argument("--backward-speed-penalty-scale", type=float, default=0.6)
    parser.add_argument("--edge-penalty-scale", type=float, default=25.0)
    parser.add_argument("--heading-penalty-scale", type=float, default=10.0)
    parser.add_argument("--speed-edge-penalty-scale", type=float, default=8.0)
    parser.add_argument("--posture-penalty-scale", type=float, default=18.0)
    parser.add_argument("--speed-posture-penalty-scale", type=float, default=10.0)
    parser.add_argument("--unsafe-reward-gate", type=float, default=0.55)
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser.parse_args()


def parse_locked_sectors(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    sectors: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value.strip()) for value in part.split("-", 1))
            sectors.extend(range(start, end + 1))
        else:
            sectors.append(int(part))
    return tuple(dict.fromkeys(sectors))


def make_env(args: argparse.Namespace) -> SectorResidualEnv:
    return SectorResidualEnv(
        host=args.host,
        port=args.port,
        config=load_controller_config(args.base_config),
        profile=load_speed_profile(args.base_profile),
        reference=load_sector_reference(args.reference),
        target_laps=args.target_laps,
        max_frames=args.max_frames,
        speed_delta=args.speed_delta,
        brake_delta_kmh=args.brake_delta_kmh,
        enable_brake_delta=args.enable_brake_delta,
        locked_sectors=parse_locked_sectors(args.locked_sectors),
        max_speed_multiplier=args.max_speed_multiplier,
        max_brake_target_speed=args.max_brake_target_speed,
        offtrack_speed_penalty_scale=args.offtrack_speed_penalty_scale,
        backward_speed_penalty_scale=args.backward_speed_penalty_scale,
        edge_penalty_scale=args.edge_penalty_scale,
        heading_penalty_scale=args.heading_penalty_scale,
        speed_edge_penalty_scale=args.speed_edge_penalty_scale,
        posture_penalty_scale=args.posture_penalty_scale,
        speed_posture_penalty_scale=args.speed_posture_penalty_scale,
        unsafe_reward_gate=args.unsafe_reward_gate,
    )


def main() -> None:
    args = parse_args()
    sys.argv = [sys.argv[0]]

    model = load_sac(args.model)
    env = make_env(args)
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            last_info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                last_info = info
                print(
                    "episode={episode} sector={sector} frames={frames} dist={dist:.1f} "
                    "reward={reward:.2f} mult={mult:.3f} brake_target={brake:.1f} reason={reason}".format(
                        episode=episode,
                        sector=info.get("sector"),
                        frames=info.get("frame_count"),
                        dist=info.get("dist_raced", 0.0),
                        reward=reward,
                        mult=info.get("speed_multiplier", 0.0),
                        brake=info.get("brake_target_speed", 0.0),
                        reason=info.get("reason"),
                    )
                )
                done = terminated or truncated
            print(
                "episode={episode} finished reason={reason} total_reward={reward:.1f} "
                "frames={frames} lap_time={lap:.3f} dist={dist:.1f}".format(
                    episode=episode,
                    reason=last_info.get("reason"),
                    reward=total_reward,
                    frames=last_info.get("frame_count"),
                    lap=last_info.get("lap_time", 0.0),
                    dist=last_info.get("dist_raced", 0.0),
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
