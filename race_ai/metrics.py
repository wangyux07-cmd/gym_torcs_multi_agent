from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EPISODE_FIELDS = [
    "run_id",
    "episode",
    "started_at",
    "ended_at",
    "wall_time_sec",
    "steps",
    "termination_reason",
    "lap_complete",
    "best_lap_time",
    "last_lap_time",
    "max_dist_raced",
    "final_dist_raced",
    "max_dist_from_start",
    "final_dist_from_start",
    "mean_speed_x",
    "max_speed_x",
    "mean_abs_track_pos",
    "max_abs_track_pos",
    "mean_abs_angle",
    "mean_steer",
    "mean_abs_steer",
    "mean_accel",
    "mean_brake",
    "max_damage",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EpisodeMetrics:
    run_id: str
    episode: int
    started_at: str = field(default_factory=utc_now)
    started_wall: float = field(default_factory=time.time)
    steps: int = 0
    max_dist_raced: float = 0.0
    final_dist_raced: float = 0.0
    max_dist_from_start: float = 0.0
    final_dist_from_start: float = 0.0
    speed_sum: float = 0.0
    max_speed: float = 0.0
    abs_track_pos_sum: float = 0.0
    max_abs_track_pos: float = 0.0
    abs_angle_sum: float = 0.0
    steer_sum: float = 0.0
    abs_steer_sum: float = 0.0
    accel_sum: float = 0.0
    brake_sum: float = 0.0
    max_damage: float = 0.0
    best_lap_time: float = 0.0
    last_lap_time: float = 0.0
    lap_complete: bool = False

    def observe(self, sensors: Mapping[str, Any], action: Mapping[str, float]) -> None:
        self.steps += 1
        dist_raced = float(sensors.get("distRaced", 0.0))
        dist_from_start = float(sensors.get("distFromStart", 0.0))
        speed_x = float(sensors.get("speedX", 0.0))
        track_pos = abs(float(sensors.get("trackPos", 0.0)))
        angle = abs(float(sensors.get("angle", 0.0)))
        damage = float(sensors.get("damage", 0.0))
        last_lap = float(sensors.get("lastLapTime", 0.0))

        self.final_dist_raced = dist_raced
        self.max_dist_raced = max(self.max_dist_raced, dist_raced)
        self.final_dist_from_start = dist_from_start
        self.max_dist_from_start = max(self.max_dist_from_start, dist_from_start)
        self.speed_sum += speed_x
        self.max_speed = max(self.max_speed, speed_x)
        self.abs_track_pos_sum += track_pos
        self.max_abs_track_pos = max(self.max_abs_track_pos, track_pos)
        self.abs_angle_sum += angle
        self.max_damage = max(self.max_damage, damage)
        if last_lap > 0.0:
            self.last_lap_time = last_lap
            self.best_lap_time = last_lap if self.best_lap_time <= 0.0 else min(self.best_lap_time, last_lap)
            self.lap_complete = True

        steer = float(action.get("steer", 0.0))
        accel = float(action.get("accel", 0.0))
        brake = float(action.get("brake", 0.0))
        self.steer_sum += steer
        self.abs_steer_sum += abs(steer)
        self.accel_sum += accel
        self.brake_sum += brake

    def finish(self, termination_reason: str) -> dict[str, Any]:
        denom = self.steps if self.steps else 1
        row = {
            "run_id": self.run_id,
            "episode": self.episode,
            "started_at": self.started_at,
            "ended_at": utc_now(),
            "wall_time_sec": round(time.time() - self.started_wall, 3),
            "steps": self.steps,
            "termination_reason": termination_reason,
            "lap_complete": self.lap_complete,
            "best_lap_time": self.best_lap_time,
            "last_lap_time": self.last_lap_time,
            "max_dist_raced": self.max_dist_raced,
            "final_dist_raced": self.final_dist_raced,
            "max_dist_from_start": self.max_dist_from_start,
            "final_dist_from_start": self.final_dist_from_start,
            "mean_speed_x": self.speed_sum / denom,
            "max_speed_x": self.max_speed,
            "mean_abs_track_pos": self.abs_track_pos_sum / denom,
            "max_abs_track_pos": self.max_abs_track_pos,
            "mean_abs_angle": self.abs_angle_sum / denom,
            "mean_steer": self.steer_sum / denom,
            "mean_abs_steer": self.abs_steer_sum / denom,
            "mean_accel": self.accel_sum / denom,
            "mean_brake": self.brake_sum / denom,
            "max_damage": self.max_damage,
        }
        return row


class RunLogger:
    def __init__(self, run_dir: str | Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.csv_path = self.run_dir / "episode_metrics.csv"
        self.jsonl_path = self.run_dir / "episode_metrics.jsonl"
        self._has_header = self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def log_episode(self, row: Mapping[str, Any]) -> None:
        clean = {key: row.get(key, "") for key in EPISODE_FIELDS}
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=EPISODE_FIELDS)
            if not self._has_header:
                writer.writeheader()
                self._has_header = True
            writer.writerow(clean)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(clean, ensure_ascii=False) + "\n")

