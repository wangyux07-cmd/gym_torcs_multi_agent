from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import snakeoil3_gym as snakeoil3

from race_ai.config_io import load_controller_config
from race_ai.controller import RuleBasedDriver
from race_ai.metrics import EpisodeMetrics, RunLogger
from race_ai.speed_profile import load_speed_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the rule driver with a distance-based speed profile.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--config", default="configs/rule_fast.json")
    parser.add_argument("--profile", default="configs/speed_profile_faster.json")
    parser.add_argument("--run-name", default="profile_driver")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--damage-limit", type=float, default=5000.0)
    parser.add_argument("--stop-on-offtrack", action="store_true")
    parser.add_argument("--stuck-steps-limit", type=int, default=300)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--telemetry", action="store_true")
    return parser.parse_args()


def make_run_id(run_name: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"


def should_stop_episode(sensors: dict, args: argparse.Namespace, completed_laps: int) -> str | None:
    if completed_laps >= args.target_laps:
        return "target_laps"
    if float(sensors.get("damage", 0.0)) >= args.damage_limit:
        return "damage_limit"
    if args.stop_on_offtrack and abs(float(sensors.get("trackPos", 0.0))) > 1.0:
        return "offtrack"
    return None


def telemetry_writer(run_dir: Path, episode: int):
    path = run_dir / f"telemetry_episode_{episode}.csv"
    fh = path.open("w", newline="", encoding="utf-8")
    fields = [
        "step",
        "dist_raced",
        "dist_from_start",
        "speed_x",
        "track_pos",
        "angle",
        "steer",
        "accel",
        "brake",
        "gear",
        "multiplier",
        "damage",
    ]
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    return fh, writer


def run_episode(
    client: snakeoil3.Client,
    driver: RuleBasedDriver,
    profile,
    logger: RunLogger,
    episode: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    metrics = EpisodeMetrics(logger.run_id, episode)
    driver.reset()
    previous_lap_time = 0.0
    completed_laps = 0
    best_progress = float("-inf")
    best_progress_step = 0
    telemetry_handle = None
    telemetry = None
    if args.telemetry:
        telemetry_handle, telemetry = telemetry_writer(run_dir, episode)

    try:
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
            base_action = driver.act(sensors, previous_accel=previous_accel)
            multiplier = profile.safe_multiplier(sensors, base_action.steer)
            action = driver.act_with_steer(
                sensors,
                steer=base_action.steer,
                previous_accel=previous_accel,
                target_speed_multiplier=multiplier,
            )

            client.R.d["steer"] = action.steer
            client.R.d["accel"] = action.accel
            client.R.d["brake"] = action.brake
            client.R.d["gear"] = action.gear
            client.R.d["meta"] = 0
            metrics.observe(sensors, client.R.d)

            if telemetry is not None:
                telemetry.writerow(
                    {
                        "step": step,
                        "dist_raced": dist_raced,
                        "dist_from_start": float(sensors.get("distFromStart", 0.0)),
                        "speed_x": float(sensors.get("speedX", 0.0)),
                        "track_pos": float(sensors.get("trackPos", 0.0)),
                        "angle": float(sensors.get("angle", 0.0)),
                        "steer": action.steer,
                        "accel": action.accel,
                        "brake": action.brake,
                        "gear": action.gear,
                        "multiplier": multiplier,
                        "damage": float(sensors.get("damage", 0.0)),
                    }
                )

            if args.print_every and step % args.print_every == 0:
                print(
                    "episode={episode} step={step} speed={speed:.1f} dist={dist:.1f} "
                    "lap={lap:.3f} steer={steer:.3f} accel={accel:.3f} brake={brake:.3f} mult={mult:.3f}".format(
                        episode=episode,
                        step=step,
                        speed=float(sensors.get("speedX", 0.0)),
                        dist=dist_raced,
                        lap=lap_time,
                        steer=action.steer,
                        accel=action.accel,
                        brake=action.brake,
                        mult=multiplier,
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
    finally:
        if telemetry_handle is not None:
            telemetry_handle.close()


def main() -> None:
    args = parse_args()
    sys.argv = [sys.argv[0]]

    config = load_controller_config(args.config)
    profile = load_speed_profile(args.profile)
    driver = RuleBasedDriver(config)
    run_id = make_run_id(args.run_name)
    run_dir = Path(args.runs_dir) / run_id
    logger = RunLogger(run_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {"driver": config.to_dict(), "speed_profile": profile.to_dict(), "args": vars(args)},
            fh,
            indent=2,
            sort_keys=True,
        )

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print("Start TORCS with scr_server waiting before running this script.")
    client = snakeoil3.Client(H=args.host, p=args.port)
    try:
        for episode in range(args.episodes):
            run_episode(client, driver, profile, logger, episode, args, run_dir)
            client.R.d["meta"] = 1
            client.respond_to_server()
            if episode < args.episodes - 1:
                time.sleep(1.0)
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
