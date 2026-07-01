from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config_io import load_controller_config

DEFAULT_DELTA_SCALE = (0.05, 0.08, 0.08)


def load_training_dependencies():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from .residual_env import ResidualDriverEnv
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3 is required for residual RL training. Install it with:\n"
            "  python -m pip install stable-baselines3 gymnasium torch tensorboard\n"
            f"Original error: {exc}"
        ) from exc
    return PPO, CheckpointCallback, Monitor, ResidualDriverEnv


def parse_delta_scale(text: str) -> tuple[float, float, float]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("delta scale must contain exactly three comma-separated floats")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a residual PPO policy on top of RuleBasedDriver.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--base-config", default="configs/rule_fast.json")
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--damage-limit", type=float, default=5000.0)
    parser.add_argument("--stop-on-offtrack", action="store_true", default=True)
    parser.add_argument("--stuck-steps-limit", type=int, default=300)
    parser.add_argument("--backward-angle-limit", type=float, default=1.25)
    parser.add_argument("--backward-speed-limit", type=float, default=-1.0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--checkpoint-freq", type=int, default=5000)
    parser.add_argument(
        "--action-mode",
        choices=("speed_target", "slow_only", "speed_only", "full"),
        default="speed_target",
        help="speed_target lets PPO adjust only the rule controller's target speed multiplier.",
    )
    parser.add_argument("--delta-scale", type=parse_delta_scale, default=DEFAULT_DELTA_SCALE)
    parser.add_argument("--min-speed-multiplier", type=float, default=0.90)
    parser.add_argument("--max-speed-multiplier", type=float, default=1.08)
    parser.add_argument("--multiplier-step-limit", type=float, default=0.01)
    parser.add_argument("--caution-speed-multiplier", type=float, default=0.98)
    parser.add_argument("--boost-start-distance", type=float, default=650.0)
    parser.add_argument("--boost-deadband", type=float, default=0.25)
    parser.add_argument("--lap-bonus", type=float, default=250.0)
    parser.add_argument("--offtrack-penalty", type=float, default=150.0)
    parser.add_argument("--damage-penalty-scale", type=float, default=1.0)
    parser.add_argument("--residual-penalty-scale", type=float, default=0.02)
    parser.add_argument("--heading-penalty-scale", type=float, default=0.35)
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=-3.0,
        help="Initial log std of the action Gaussian. Lower = smaller initial corrections, "
        "so early random exploration doesn't constantly saturate the delta range and crash.",
    )
    parser.add_argument("--out-dir", default="models/residual_ppo")
    parser.add_argument("--resume-from", default=None, help="Path to an existing model .zip to continue training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PPO, CheckpointCallback, Monitor, ResidualDriverEnv = load_training_dependencies()

    # snakeoil parses sys.argv internally; keep our CLI flags away from it.
    sys.argv = [sys.argv[0]]

    base_config = load_controller_config(args.base_config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = ResidualDriverEnv(
        host=args.host,
        port=args.port,
        config=base_config,
        target_laps=args.target_laps,
        max_steps=args.max_steps,
        damage_limit=args.damage_limit,
        stop_on_offtrack=args.stop_on_offtrack,
        stuck_steps_limit=args.stuck_steps_limit,
        backward_angle_limit=args.backward_angle_limit,
        backward_speed_limit=args.backward_speed_limit,
        action_mode=args.action_mode,
        delta_scale=args.delta_scale,
        min_speed_multiplier=args.min_speed_multiplier,
        max_speed_multiplier=args.max_speed_multiplier,
        multiplier_step_limit=args.multiplier_step_limit,
        caution_speed_multiplier=args.caution_speed_multiplier,
        boost_start_distance=args.boost_start_distance,
        boost_deadband=args.boost_deadband,
        lap_bonus=args.lap_bonus,
        offtrack_penalty=args.offtrack_penalty,
        damage_penalty_scale=args.damage_penalty_scale,
        residual_penalty_scale=args.residual_penalty_scale,
        heading_penalty_scale=args.heading_penalty_scale,
    )
    env = Monitor(
        env,
        filename=str(out_dir / "monitor.csv"),
        info_keywords=(
            "reason",
            "dist_raced",
            "best_progress",
            "lap_time",
            "speed_multiplier",
            "mean_speed_multiplier",
            "max_speed_multiplier",
            "track_pos",
            "angle",
            "speed_x",
            "progress",
        ),
    )

    if args.resume_from:
        model = PPO.load(args.resume_from, env=env)
        print(f"Resumed model from {args.resume_from}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            policy_kwargs={"log_std_init": args.log_std_init},
            verbose=1,
            tensorboard_log=str(out_dir / "tb"),
        )

    with (out_dir / "training_config.json").open("w", encoding="utf-8") as fh:
        payload = vars(args).copy()
        payload["delta_scale"] = list(args.delta_scale)
        payload["base_config_values"] = base_config.to_dict()
        json.dump(payload, fh, indent=2, sort_keys=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="residual_ppo",
    )

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback, reset_num_timesteps=args.resume_from is None)
    finally:
        model.save(str(out_dir / "final_model"))
        env.close()
        print(f"Saved final model to {out_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()
