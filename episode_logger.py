"""
episode_logger.py
=================
Per-episode CSV logging for thesis data collection.

This module owns all CSV file lifecycle (open / write / close) that
previously lived inside TorcsEnv.  Extracting it here means:

  - Changing the CSV schema (adding/removing columns) is a one-file change.
  - The schema is defined in a single list (LOG_COLUMNS) -- adding a new
    reward term requires editing reward.py's info dict AND LOG_COLUMNS here;
    nowhere else.
  - TorcsEnv becomes testable without file I/O: just don't create an
    EpisodeLogger, or pass logging_enabled=False.

Usage
-----
    logger = EpisodeLogger(episode_count=1)
    logger.log_step(time_step, raw_obs, action, reward, reward_info,
                    gear, terminal_reason)
    logger.close()
"""

import csv
import os
from datetime import datetime
from typing import Dict, Optional

import config


# Column order for every CSV row.  Must stay in sync with _build_row().
LOG_COLUMNS = [
    "step",
    # Raw sensor readings
    "speedX", "speedY", "angle", "trackPos",
    "distFromStart", "distRaced", "damage",
    # Agent actions
    "steer", "accel", "brake", "gear",
    # Reward breakdown
    "r_core_progress", "r_safety", "r_smooth", "r_anticipation",
    "r_time", "r_terminal", "r_checkpoint", "r_record", "reward_total",
    # Episode event
    "terminal_reason",
]


class EpisodeLogger:
    """
    Manages a single CSV file for one episode.

    Create a new instance at the start of each episode; call close() (or
    use it as a context manager) at the end.
    """

    def __init__(self, episode_count: int):
        if not config.LOGGING["enabled"]:
            self._writer = None
            self._file   = None
            return

        log_dir   = config.LOGGING["log_dir"]
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"ep{episode_count:05d}_{timestamp}.csv"
        filepath  = os.path.join(log_dir, filename)

        self._file   = open(filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(LOG_COLUMNS)

    # ------------------------------------------------------------------

    def log_step(
        self,
        time_step: int,
        raw_obs: dict,
        action,               # numpy array shape (3,): [steer, accel, brake]
        reward: float,
        reward_info: Dict[str, float],
        gear: int,
        terminal_reason: str,
    ) -> None:
        """Write one data row.  No-op if logging is disabled."""
        if self._writer is None:
            return
        self._writer.writerow(self._build_row(
            time_step, raw_obs, action, reward, reward_info, gear, terminal_reason
        ))

    def log_training_stopped(self, time_step: int) -> None:
        """
        Write a sentinel row when training ends mid-episode (not a real
        terminal condition).  Prevents CSV files from silently truncating.
        """
        if self._writer is None:
            return
        row = [time_step] + [""] * (len(LOG_COLUMNS) - 2) + ["training_stopped"]
        self._writer.writerow(row)

    def close(self) -> None:
        """Flush and close the underlying file."""
        if self._file is not None:
            self._file.close()
            self._file   = None
            self._writer = None

    # ------------------------------------------------------------------
    # Context-manager support (optional convenience)
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------

    @staticmethod
    def _build_row(
        time_step: int,
        raw_obs: dict,
        action,
        reward: float,
        reward_info: Dict[str, float],
        gear: int,
        terminal_reason: str,
    ) -> list:
        return [
            time_step,
            raw_obs.get("speedX",        0),
            raw_obs.get("speedY",        0),
            raw_obs.get("angle",         0),
            raw_obs.get("trackPos",      0),
            raw_obs.get("distFromStart", 0),
            raw_obs.get("distRaced",     0),
            raw_obs.get("damage",        0),
            float(action[0]),   # steer
            float(action[1]),   # accel
            float(action[2]),   # brake
            gear,
            reward_info.get("r_core_progress", 0),
            reward_info.get("r_safety",        0),
            reward_info.get("r_smooth",        0),
            reward_info.get("r_anticipation",  0),
            reward_info.get("r_time",          0),
            reward_info.get("terminal",        0),
            reward_info.get("checkpoint",      0),
            reward_info.get("record",          0),
            reward,
            terminal_reason,
        ]
