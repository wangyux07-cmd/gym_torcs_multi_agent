from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
from datetime import datetime
from pathlib import Path

import snakeoil3_gym as snakeoil3

from race_ai.config_io import load_controller_config
from race_ai.controller import RuleBasedDriver, clip
from race_ai.metrics import EpisodeMetrics, RunLogger

DEFAULT_DELTA_SCALE = (0.05, 0.08, 0.08)


def parse_delta_scale(text: str) -> tuple[float, float, float]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("delta scale must contain exactly three comma-separated floats")
    return values


def load_ppo_model(path: str) -> Any:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3 is required to run a residual PPO model. Install it with:\n"
            "  python -m pip install stable-baselines3 gymnasium torch tensorboard\n"
            f"Original error: {exc}"
        ) from exc
    return PPO.load(path)


def load_runtime_dependencies():
    try:
        import numpy as np
        from race_ai.residual_observation import build_observation
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "numpy is required to run residual inference. Install dependencies with:\n"
            "  python -m pip install -r requirements-rl.txt\n"
            f"Original error: {exc}"
        ) from exc
    return np, build_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RuleBasedDriver + a trained residual PPO correction.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--config", default="configs/rule_fast.json")
    parser.add_argument("--model", default="models/residual_ppo/final_model.zip")
    parser.add_argument(
        "--action-mode",
        choices=("speed_target", "slow_only", "speed_only", "full"),
        default="speed_target",
        help="Must match training. speed_target applies PPO output as a target speed multiplier.",
    )
    parser.add_argument("--delta-scale", type=parse_delta_scale, default=DEFAULT_DELTA_SCALE)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--min-speed-multiplier", type=float, default=0.90)
    parser.add_argument("--max-speed-multiplier", type=float, default=1.08)
    parser.add_argument("--multiplier-step-limit", type=float, default=0.01)
    parser.add_argument("--caution-speed-multiplier", type=float, default=0.98)
    parser.add_argument("--boost-start-distance", type=float, default=650.0)
    parser.add_argument("--boost-deadband", type=float, default=0.25)
    parser.add_argument("--run-name", default="residual_driver")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--damage-limit", type=float, default=5000.0)
    parser.add_argument("--stop-on-offtrack", action="store_true")
    parser.add_argument("--stuck-steps-limit", type=int, default=300)
    parser.add_argument("--backward-angle-limit", type=float, default=1.25)
    parser.add_argument("--backward-speed-limit", type=float, default=-1.0)
    parser.add_argument("--print-every", type=int, default=250)
    return parser.parse_args()


def make_run_id(run_name: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"


def should_stop_episode(sensors: dict, args: argparse.Namespace, completed_laps: int) -> str | None:
    if completed_laps >= args.target_laps:
        return "target_laps"
    if float(sensors.get("damage", 0.0)) >= args.damage_limit:
        return "damage_limit"
    if abs(float(sensors.get("angle", 0.0))) > args.backward_angle_limit:
        return "backward"
    if float(sensors.get("speedX", 0.0)) < args.backward_speed_limit:
        return "backward"
    if args.stop_on_offtrack:
        if abs(float(sensors.get("trackPos", 0.0))) > 1.0:
            return "offtrack"
    return None


def safe_speed_multiplier(sensors: dict, rule_steer: float, previous: float, raw_action: float, args: argparse.Namespace) -> float:
    dist = float(sensors.get("distRaced", sensors.get("distFromStart", 0.0)))
    if dist < args.boost_start_distance:
        desired = 1.0
    else:
        boost_signal = max(0.0, float(raw_action) - args.boost_deadband) / max(1e-6, 1.0 - args.boost_deadband)
        desired = 1.0 + boost_signal * (args.max_speed_multiplier - 1.0) * float(args.residual_scale)
    desired = clip(desired, args.min_speed_multiplier, args.max_speed_multiplier)

    track = list(sensors.get("track", [200.0] * 19))
    ahead = float(track[9])
    side_clearance = min(float(v) for v in track[4:15])
    safety_cap = args.max_speed_multiplier
    angle_abs = abs(float(sensors.get("angle", 0.0)))
    track_pos_abs = abs(float(sensors.get("trackPos", 0.0)))
    caution_cap = max(args.min_speed_multiplier, min(1.0, args.caution_speed_multiplier))
    if angle_abs > 0.55 or track_pos_abs > 0.72:
        safety_cap = caution_cap
    elif ahead < 45.0 or side_clearance < 14.0 or abs(rule_steer) > 0.28 or angle_abs > 0.36 or track_pos_abs > 0.55:
        safety_cap = 1.0
    elif ahead < 75.0 or side_clearance < 24.0 or abs(rule_steer) > 0.18 or angle_abs > 0.24 or track_pos_abs > 0.38:
        safety_cap = min(safety_cap, 1.04)

    desired = min(desired, safety_cap)
    if desired < previous:
        return clip(desired, args.min_speed_multiplier, previous)
    upper = min(args.max_speed_multiplier, previous + args.multiplier_step_limit)
    return clip(desired, previous, upper)


def run_episode(
    client: snakeoil3.Client,
    driver: RuleBasedDriver,
    model: Any,
    logger: RunLogger,
    episode: int,
    args: argparse.Namespace,
    np: Any,
    build_observation: Any,
) -> None:
    metrics = EpisodeMetrics(logger.run_id, episode)
    driver.reset()
    previous_lap_time = 0.0
    completed_laps = 0
    best_progress = float("-inf")
    best_progress_step = 0
    speed_multiplier = 1.0

    for step in range(args.max_steps):
        client.get_servers_input()
        sensors = client.S.d
        lap_time = float(sensors.get("lastLapTime", 0.0))
        if lap_time > 0.0 and lap_time != previous_lap_time:
            completed_laps += 1
            previous_lap_time = lap_time

        dist_raced = float(sensors.get("distRaced", 0.0))
        if dist_raced > best_progress:
            best_progress = dist_raced
            best_progress_step = step
        stuck = bool(args.stuck_steps_limit) and (step - best_progress_step) >= args.stuck_steps_limit

        previous_accel = float(client.R.d.get("accel", 0.0))
        rule_action = driver.act(sensors, previous_accel=previous_accel, target_speed_multiplier=speed_multiplier)
        obs = build_observation(sensors, rule_action)
        rl_action, _ = model.predict(obs, deterministic=True)
        rl_action = np.clip(rl_action, -1.0, 1.0).astype(np.float32)
        if args.action_mode == "speed_target":
            speed_multiplier = safe_speed_multiplier(sensors, rule_action.steer, speed_multiplier, float(rl_action[0]), args)
            rule_action = driver.act_with_steer(
                sensors,
                steer=rule_action.steer,
                previous_accel=previous_accel,
                target_speed_multiplier=speed_multiplier,
            )
            delta = np.zeros(3, dtype=np.float32)
        else:
            delta_scale = np.asarray(args.delta_scale, dtype=np.float32) * float(args.residual_scale)
        if args.action_mode == "speed_target":
            pass
        elif args.action_mode == "slow_only":
            delta = np.array(
                [
                    0.0,
                    min(0.0, float(rl_action[0])) * delta_scale[1],
                    max(0.0, float(rl_action[1])) * delta_scale[2],
                ],
                dtype=np.float32,
            )
        elif args.action_mode == "speed_only":
            delta = np.array([0.0, rl_action[0] * delta_scale[1], rl_action[1] * delta_scale[2]], dtype=np.float32)
        else:
            delta = rl_action * delta_scale

        final_steer = clip(rule_action.steer + float(delta[0]), -1.0, 1.0)
        final_accel = clip(rule_action.accel + float(delta[1]), 0.0, 1.0)
        final_brake = clip(rule_action.brake + float(delta[2]), 0.0, 1.0)
        if final_brake > 0.0:
            final_accel = 0.0

        client.R.d["steer"] = final_steer
        client.R.d["accel"] = final_accel
        client.R.d["brake"] = final_brake
        client.R.d["gear"] = rule_action.gear
        client.R.d["meta"] = 0
        metrics.observe(sensors, client.R.d)

        if args.print_every and step % args.print_every == 0:
            print(
                "episode={episode} step={step} speed={speed:.1f} dist={dist:.1f} "
                "lap={lap:.3f} steer={steer:.3f} accel={accel:.3f} brake={brake:.3f} mult={mult:.3f}".format(
                    episode=episode,
                    step=step,
                    speed=float(sensors.get("speedX", 0.0)),
                    dist=dist_raced,
                    lap=lap_time,
                    steer=final_steer,
                    accel=final_accel,
                    brake=final_brake,
                    mult=speed_multiplier,
                )
            )

        reason = should_stop_episode(sensors, args, completed_laps) or ("stuck" if stuck else None)
        client.respond_to_server()
        if reason:
            row = metrics.finish(reason)
            logger.log_episode(row)
            print(
                "episode={episode} finished reason={reason} steps={steps} best_lap={best:.3f} "
                "max_dist={dist:.1f} max_speed={speed:.1f}".format(
                    episode=episode,
                    reason=reason,
                    steps=row["steps"],
                    best=row["best_lap_time"],
                    dist=row["max_dist_raced"],
                    speed=row["max_speed_x"],
                )
            )
            return

    row = metrics.finish("max_steps")
    logger.log_episode(row)
    print(
        "episode={episode} finished reason=max_steps steps={steps} best_lap={best:.3f} max_dist={dist:.1f}".format(
            episode=episode,
            steps=row["steps"],
            best=row["best_lap_time"],
            dist=row["max_dist_raced"],
        )
    )


def main() -> None:
    args = parse_args()
    np, build_observation = load_runtime_dependencies()

    # snakeoil parses sys.argv internally; keep user CLI flags away from it.
    sys.argv = [sys.argv[0]]

    config = load_controller_config(args.config)
    driver = RuleBasedDriver(config)
    model = load_ppo_model(args.model)
    run_id = make_run_id(args.run_name)
    run_dir = Path(args.runs_dir) / run_id
    logger = RunLogger(run_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump({"driver": config.to_dict(), "model": args.model, "args": vars(args)}, fh, indent=2, sort_keys=True)

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print("Start TORCS with scr_server waiting before running this script.")
    client = snakeoil3.Client(H=args.host, p=args.port)
    try:
        for episode in range(args.episodes):
            run_episode(client, driver, model, logger, episode, args, np, build_observation)
            client.R.d["meta"] = 1
            client.respond_to_server()
            if episode < args.episodes - 1:
                time.sleep(1.0)
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
