from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config_io import load_controller_config
from .sector_reference import load_sector_reference
from .speed_profile import load_speed_profile


def load_training_dependencies():
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor

        from .sector_residual_env import SectorResidualEnv
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3 is required for sector SAC training. Install it with:\n"
            "  python -m pip install stable-baselines3 gymnasium torch tensorboard\n"
            f"Original error: {exc}"
        ) from exc
    return SAC, BaseCallback, CallbackList, CheckpointCallback, Monitor, SectorResidualEnv


SECTOR_TELEMETRY_FIELDS = [
    "env_step",
    "episode",
    "episode_step",
    "reward",
    "done",
    "reason",
    "sector",
    "sector_steps",
    "reference_sector_steps",
    "frame_count",
    "dist_raced",
    "lap_time",
    "speed_x",
    "track_pos",
    "angle",
    "speed_multiplier",
    "sector_entry_cap",
    "brake_target_speed",
    "speed_delta_action",
    "brake_delta_action",
    "max_abs_track_pos",
    "max_abs_angle",
    "max_speed_x",
    "edge_risk",
    "heading_risk",
    "speed_edge_risk",
    "posture_risk",
    "speed_posture_risk",
    "unsafe_fraction",
    "safety_gate",
    "sector_time_delta",
    "reward_sector_time",
    "reward_progress",
    "penalty_edge",
    "penalty_heading",
    "penalty_speed_edge",
    "penalty_posture",
    "penalty_speed_posture",
]


def make_sector_telemetry_callback(base_cls, path: Path):
    class SectorTelemetryCallback(base_cls):
        def __init__(self):
            super().__init__()
            self.path = path
            self.handle = None
            self.writer = None
            self.episode = 0
            self.episode_step = 0

        def _on_training_start(self) -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.handle, fieldnames=SECTOR_TELEMETRY_FIELDS)
            self.writer.writeheader()

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            rewards = self.locals.get("rewards", [])
            dones = self.locals.get("dones", [])
            reserved_fields = {"env_step", "episode", "episode_step", "reward", "done"}
            for index, info in enumerate(infos):
                row = {field: "" for field in SECTOR_TELEMETRY_FIELDS}
                row["env_step"] = self.num_timesteps
                row["episode"] = self.episode
                row["episode_step"] = self.episode_step
                row["reward"] = float(rewards[index]) if index < len(rewards) else ""
                row["done"] = bool(dones[index]) if index < len(dones) else False
                for field in SECTOR_TELEMETRY_FIELDS:
                    if field not in reserved_fields and field in info:
                        row[field] = info[field]
                assert self.writer is not None
                self.writer.writerow(row)
                if row["done"]:
                    self.episode += 1
                    self.episode_step = 0
                else:
                    self.episode_step += 1
            if self.handle is not None:
                self.handle.flush()
            return True

        def _on_training_end(self) -> None:
            if self.handle is not None:
                self.handle.close()
                self.handle = None

    return SectorTelemetryCallback()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sector-level SAC residuals on top of the stable TORCS profile driver.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--base-config", default="configs/rule_fast.json")
    parser.add_argument("--base-profile", default="configs/speed_profile_v2_12_finish_brake_170.json")
    parser.add_argument("--reference", default="configs/sector_reference_v2_12_50m.json")
    parser.add_argument("--out-dir", default="models/sector_sac_speed_only")
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument("--damage-limit", type=float, default=5000.0)
    parser.add_argument("--stuck-frames-limit", type=int, default=300)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto_0.1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-freq", type=int, default=1000)
    parser.add_argument("--speed-delta", type=float, default=0.04)
    parser.add_argument("--brake-delta-kmh", type=float, default=5.0)
    parser.add_argument(
        "--enable-brake-delta",
        action="store_true",
        help="Allow SAC to adjust the brake target. Disabled by default because the first run should be speed-only.",
    )
    parser.add_argument(
        "--locked-sectors",
        type=parse_locked_sectors,
        default=tuple(range(44, 51)) + tuple(range(61, 67)),
        help="Comma-separated sector indexes/ranges forced to baseline residual action, e.g. '44-50,61-66'.",
    )
    parser.add_argument("--min-speed-multiplier", type=float, default=0.92)
    parser.add_argument("--max-speed-multiplier", type=float, default=1.44)
    parser.add_argument("--min-brake-target-speed", type=float, default=120.0)
    parser.add_argument("--max-brake-target-speed", type=float, default=170.0)
    parser.add_argument(
        "--sector-step-reward",
        type=float,
        default=8.0,
        help="Scale for normalized sector-time improvement: (reference_steps - actual_steps) / reference_steps.",
    )
    parser.add_argument(
        "--progress-reward",
        type=float,
        default=0.0,
        help="Optional raw progress reward. Default is 0 because the baseline already finishes the lap.",
    )
    parser.add_argument("--finish-bonus", type=float, default=300.0)
    parser.add_argument("--finish-step-reward", type=float, default=0.08)
    parser.add_argument("--failure-penalty", type=float, default=240.0)
    parser.add_argument("--offtrack-speed-penalty-scale", type=float, default=0.9)
    parser.add_argument("--backward-speed-penalty-scale", type=float, default=0.6)
    parser.add_argument("--edge-penalty-scale", type=float, default=25.0)
    parser.add_argument("--heading-penalty-scale", type=float, default=10.0)
    parser.add_argument("--speed-edge-penalty-scale", type=float, default=8.0)
    parser.add_argument("--posture-penalty-scale", type=float, default=18.0)
    parser.add_argument("--speed-posture-penalty-scale", type=float, default=10.0)
    parser.add_argument("--unsafe-reward-gate", type=float, default=0.55)
    parser.add_argument("--action-penalty-scale", type=float, default=0.03)
    parser.add_argument("--action-change-penalty-scale", type=float, default=0.12)
    parser.add_argument("--resume-from", default=None)
    return parser.parse_args()


def make_env(args: argparse.Namespace):
    _, _, _, _, Monitor, SectorResidualEnv = load_training_dependencies()
    env = SectorResidualEnv(
        host=args.host,
        port=args.port,
        config=load_controller_config(args.base_config),
        profile=load_speed_profile(args.base_profile),
        reference=load_sector_reference(args.reference),
        target_laps=args.target_laps,
        max_frames=args.max_frames,
        damage_limit=args.damage_limit,
        stuck_frames_limit=args.stuck_frames_limit,
        speed_delta=args.speed_delta,
        brake_delta_kmh=args.brake_delta_kmh,
        enable_brake_delta=args.enable_brake_delta,
        locked_sectors=args.locked_sectors,
        min_speed_multiplier=args.min_speed_multiplier,
        max_speed_multiplier=args.max_speed_multiplier,
        min_brake_target_speed=args.min_brake_target_speed,
        max_brake_target_speed=args.max_brake_target_speed,
        sector_step_reward=args.sector_step_reward,
        progress_reward=args.progress_reward,
        finish_bonus=args.finish_bonus,
        finish_step_reward=args.finish_step_reward,
        failure_penalty=args.failure_penalty,
        offtrack_speed_penalty_scale=args.offtrack_speed_penalty_scale,
        backward_speed_penalty_scale=args.backward_speed_penalty_scale,
        edge_penalty_scale=args.edge_penalty_scale,
        heading_penalty_scale=args.heading_penalty_scale,
        speed_edge_penalty_scale=args.speed_edge_penalty_scale,
        posture_penalty_scale=args.posture_penalty_scale,
        speed_posture_penalty_scale=args.speed_posture_penalty_scale,
        unsafe_reward_gate=args.unsafe_reward_gate,
        action_penalty_scale=args.action_penalty_scale,
        action_change_penalty_scale=args.action_change_penalty_scale,
    )
    return Monitor(
        env,
        filename=str(Path(args.out_dir) / "monitor.csv"),
        info_keywords=(
            "reason",
            "sector",
            "sector_steps",
            "reference_sector_steps",
            "frame_count",
            "dist_raced",
            "lap_time",
            "speed_x",
            "track_pos",
            "angle",
            "speed_multiplier",
            "sector_entry_cap",
            "brake_target_speed",
            "speed_delta_action",
            "brake_delta_action",
            "max_abs_track_pos",
            "max_abs_angle",
            "max_speed_x",
            "edge_risk",
            "heading_risk",
            "speed_edge_risk",
            "posture_risk",
            "speed_posture_risk",
            "unsafe_fraction",
            "safety_gate",
            "sector_time_delta",
            "reward_sector_time",
            "reward_progress",
            "penalty_edge",
            "penalty_heading",
            "penalty_speed_edge",
            "penalty_posture",
            "penalty_speed_posture",
        ),
    )


def main() -> None:
    args = parse_args()
    SAC, BaseCallback, CallbackList, CheckpointCallback, _, _ = load_training_dependencies()
    sys.argv = [sys.argv[0]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    if args.resume_from:
        model = SAC.load(args.resume_from, env=env)
        print(f"Resumed SAC model from {args.resume_from}")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            policy_kwargs={"net_arch": [128, 128], "log_std_init": -3.5},
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(out_dir / "tb"),
        )

    payload = vars(args).copy()
    payload["base_config_values"] = load_controller_config(args.base_config).to_dict()
    payload["base_profile_values"] = load_speed_profile(args.base_profile).to_dict()
    payload["reference_values"] = load_sector_reference(args.reference).to_dict()
    with (out_dir / "training_config.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="sector_sac",
    )
    telemetry_callback = make_sector_telemetry_callback(BaseCallback, out_dir / "sector_telemetry.csv")
    callbacks = CallbackList([checkpoint_callback, telemetry_callback])

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=args.resume_from is None,
        )
    finally:
        model.save(str(out_dir / "final_model"))
        env.close()
        print(f"Saved final model to {out_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
