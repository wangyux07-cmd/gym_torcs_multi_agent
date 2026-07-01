from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .results_store import RaceResult, ResultsStore, now_iso, timestamp_id


STEP_RE = re.compile(
    r"episode=(?P<episode>\d+)\s+step=(?P<step>\d+)\s+speed=(?P<speed>-?\d+(?:\.\d+)?)\s+"
    r"dist=(?P<dist>-?\d+(?:\.\d+)?)\s+lap=(?P<lap>-?\d+(?:\.\d+)?)"
)
FINISH_RE = re.compile(
    r"episode=(?P<episode>\d+)\s+finished\s+reason=(?P<reason>\S+)\s+"
    r"steps=(?P<steps>\d+)\s+best_lap=(?P<lap>-?\d+(?:\.\d+)?)\s+"
    r"max_dist=(?P<dist>-?\d+(?:\.\d+)?)"
)


@dataclass
class LiveRace:
    run_id: str = ""
    user_id: str = "default"
    username: str = "Anonymous"
    status: str = "idle"
    started_at: str | None = None
    finished_at: str | None = None
    message: str = "Ready"
    step_count: int | None = None
    speed: float | None = None
    distance: float = 0.0
    lap_time: float | None = None
    reason: str = ""
    log_tail: list[str] = field(default_factory=list)


class RaceRunner:
    def __init__(self, root: Path, results: ResultsStore) -> None:
        self.root = root
        self.results = results
        self.driver_config = "configs/rule_fast.json"
        self.process: subprocess.Popen[str] | None = None
        self.live = LiveRace()
        self.lock = threading.RLock()

    def start(self, user_id: str = "default", username: str = "Anonymous", target_laps: int = 1, config_path: str = "configs/rule_fast.json") -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return self.snapshot()
            config_full = self.root / config_path
            if not config_full.exists():
                raise FileNotFoundError(f"Driver config not found: {config_full}")
            self.driver_config = config_path
            run_id = timestamp_id()
            self.live = LiveRace(
                run_id=run_id,
                user_id=user_id,
                username=username,
                status="starting",
                started_at=now_iso(),
                message="Starting AI driver...",
            )
            command = [
                sys.executable,
                str(self.root / "race_rule_driver.py"),
                "--config", config_path,
                "--episodes", "1",
                "--target-laps", str(target_laps),
                "--stuck-steps-limit", "300",
                "--print-every", "100",
            ]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(
                command,
                cwd=str(self.root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()
            threading.Thread(target=self._watch_process, args=(self.process,), daemon=True).start()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            process = self.process
            if process is not None and process.poll() is None:
                self.live.status = "stopping"
                self.live.message = "Stopping race..."
                process.terminate()
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            payload = self.live.__dict__.copy()
            payload["running"] = self.process is not None and self.process.poll() is None
            payload["driver_config"] = self.driver_config
            payload["progress"] = round(min(100.0, max(0.0, self.live.distance / 3608.45 * 100.0)), 2)
            return payload

    def _append_log(self, line: str) -> None:
        self.live.log_tail.append(line)
        self.live.log_tail = self.live.log_tail[-24:]

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            with self.lock:
                self._append_log(line)
                lower = line.lower()
                if "waiting" in lower or "connecting" in lower:
                    if self.live.status == "starting":
                        self.live.status = "waiting"
                        self.live.message = "Waiting for TORCS simulator..."
                step = STEP_RE.search(line)
                if step:
                    self.live.status = "racing"
                    self.live.message = "AI driver is on track."
                    self.live.step_count = int(step.group("step"))
                    self.live.speed = float(step.group("speed"))
                    self.live.distance = float(step.group("dist"))
                    continue
                finish = FINISH_RE.search(line)
                if finish:
                    reason = finish.group("reason")
                    lap_time = float(finish.group("lap"))
                    self.live.status = "finished" if reason == "target_laps" else "crashed"
                    self.live.reason = reason
                    self.live.message = "Race complete." if reason == "target_laps" else "The car left the track."
                    self.live.finished_at = now_iso()
                    self.live.step_count = int(finish.group("steps"))
                    self.live.lap_time = lap_time if lap_time > 0 else None
                    self.live.distance = float(finish.group("dist"))

    def _watch_process(self, process: subprocess.Popen[str]) -> None:
        return_code = process.wait()
        with self.lock:
            if self.process is not process:
                return
            if self.live.finished_at is None:
                self.live.finished_at = now_iso()
            if self.live.status == "stopping":
                self.live.status = "stopped"
                self.live.message = "Race stopped."
            elif self.live.status in {"starting", "waiting", "racing"}:
                self.live.status = "stopped" if return_code == 0 else "error"
                self.live.message = "Race process stopped." if return_code == 0 else "Race process exited with an error."
            self._store_current_result()
            self.process = None

    def _store_current_result(self) -> None:
        if not self.live.run_id:
            return
        result = RaceResult(
            id=self.live.run_id,
            username=self.live.username,
            started_at=self.live.started_at or "",
            finished_at=self.live.finished_at or now_iso(),
            status=self.live.status,
            reason=self.live.reason,
            lap_time=self.live.lap_time,
            frame_count=self.live.step_count,
            distance=self.live.distance,
            total_reward=None,
            best_model=self.driver_config,
            log_tail=list(self.live.log_tail),
        )
        self.results.add_result(self.live.user_id, result)
