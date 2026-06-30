from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import snakeoil3_gym as snakeoil3

from race_ai.config_io import load_controller_config, save_default_config
from race_ai.controller import RuleBasedDriver
from race_ai.metrics import EpisodeMetrics, RunLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the rule-based TORCS driver.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--config", default="configs/rule_fast.json")
    parser.add_argument("--save-default-config", default=None)
    parser.add_argument("--run-name", default="rule_driver")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--damage-limit", type=float, default=5000.0)
    parser.add_argument("--stop-on-offtrack", action="store_true")
    parser.add_argument(
        "--stuck-steps-limit",
        type=int,
        default=300,
        help="Stop the episode if distRaced hasn't advanced for this many consecutive steps "
        "(car stalled or wedged against a wall). 0 disables the check.",
    )
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--vision", action="store_true")
    return parser.parse_args()


def make_run_id(run_name: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"


def should_stop_episode(sensors: dict, args: argparse.Namespace, completed_laps: int) -> str | None:
    if completed_laps >= args.target_laps:
        return "target_laps"
    if float(sensors.get("damage", 0.0)) >= args.damage_limit:
        return "damage_limit"
    if args.stop_on_offtrack:
        if abs(float(sensors.get("trackPos", 0.0))) > 1.0:
            return "offtrack"
    return None


def run_episode(client: snakeoil3.Client, driver: RuleBasedDriver, logger: RunLogger, episode: int, args: argparse.Namespace) -> None:
    metrics = EpisodeMetrics(logger.run_id, episode)
    driver.reset()
    previous_lap_time = 0.0
    completed_laps = 0
    best_progress = float("-inf")
    best_progress_step = 0

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

        action = driver.act(sensors, previous_accel=float(client.R.d.get("accel", 0.0)))
        client.R.d["steer"] = action.steer
        client.R.d["accel"] = action.accel
        client.R.d["brake"] = action.brake
        client.R.d["gear"] = action.gear
        client.R.d["meta"] = 0
        metrics.observe(sensors, client.R.d)

        if args.print_every and step % args.print_every == 0:
            print(
                "episode={episode} step={step} speed={speed:.1f} dist={dist:.1f} "
                "lap={lap:.3f} steer={steer:.3f} accel={accel:.3f} brake={brake:.3f}".format(
                    episode=episode,
                    step=step,
                    speed=float(sensors.get("speedX", 0.0)),
                    dist=float(sensors.get("distRaced", 0.0)),
                    lap=lap_time,
                    steer=action.steer,
                    accel=action.accel,
                    brake=action.brake,
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
    if args.save_default_config:
        save_default_config(args.save_default_config)
        print(f"Saved default config to {args.save_default_config}")
        return

    # snakeoil parses sys.argv internally; keep user CLI flags away from it.
    sys.argv = [sys.argv[0]]

    config = load_controller_config(args.config)
    driver = RuleBasedDriver(config)
    run_id = make_run_id(args.run_name)
    run_dir = Path(args.runs_dir) / run_id
    logger = RunLogger(run_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump({"driver": config.to_dict(), "args": vars(args)}, fh, indent=2, sort_keys=True)

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print("Start TORCS with scr_server waiting before running this script.")
    client = snakeoil3.Client(H=args.host, p=args.port, vision=args.vision)
    try:
        for episode in range(args.episodes):
            run_episode(client, driver, logger, episode, args)
            # Tell the server to end/restart the race even if our requested
            # target laps is fewer than the race config's configured laps -
            # otherwise the server is left mid-race when we disconnect.
            client.R.d["meta"] = 1
            client.respond_to_server()
            if episode < args.episodes - 1:
                time.sleep(1.0)
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
