from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .database import get_conn


@dataclass
class RaceResult:
    id: str
    username: str
    started_at: str
    finished_at: str
    status: str
    reason: str
    lap_time: float | None
    frame_count: int | None
    distance: float
    total_reward: float | None
    best_model: str
    log_tail: list[str]


class ResultsStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def list_results(self, user_id: str = "default") -> list[dict[str, Any]]:
        try:
            with get_conn(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM results WHERE user_id = ? ORDER BY started_at DESC LIMIT 200",
                    (user_id,),
                ).fetchall()
            return [_row_to_dict(row) for row in rows]
        except Exception:
            return []

    def add_result(self, user_id: str, result: RaceResult) -> None:
        try:
            with get_conn(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO results "
                    "(id, user_id, username, started_at, finished_at, status, reason, "
                    "lap_time, frame_count, distance, total_reward, best_model, log_tail) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        result.id, user_id, result.username,
                        result.started_at, result.finished_at,
                        result.status, result.reason,
                        result.lap_time, result.frame_count,
                        result.distance, result.total_reward,
                        result.best_model, json.dumps(result.log_tail),
                    ),
                )
        except Exception:
            pass

    def best_result(self, user_id: str = "default") -> dict[str, Any] | None:
        try:
            with get_conn(self.db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM results WHERE user_id = ? AND status = 'finished' "
                    "AND lap_time IS NOT NULL ORDER BY lap_time ASC LIMIT 1",
                    (user_id,),
                ).fetchone()
            return _row_to_dict(row) if row else None
        except Exception:
            return None

    def leaderboard(self, limit: int = 20) -> list[dict[str, Any]]:
        """Global best lap per user, sorted fastest first."""
        try:
            with get_conn(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT user_id, username, MIN(lap_time) AS lap_time, COUNT(*) AS races "
                    "FROM results "
                    "WHERE status = 'finished' AND lap_time IS NOT NULL "
                    "GROUP BY user_id "
                    "ORDER BY lap_time ASC "
                    "LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []

    def get_username(self, user_id: str) -> str:
        try:
            with get_conn(self.db_path) as conn:
                row = conn.execute(
                    "SELECT username FROM users WHERE user_id = ?", (user_id,)
                ).fetchone()
            return row["username"] if row else "Anonymous"
        except Exception:
            return "Anonymous"

    def set_username(self, user_id: str, username: str) -> str:
        username = username.strip()[:32] or "Anonymous"
        try:
            with get_conn(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO users (user_id, username) VALUES (?, ?)",
                    (user_id, username),
                )
        except Exception:
            pass
        return username


def _row_to_dict(row: Any) -> dict[str, Any]:
    d = dict(row)
    d.pop("user_id", None)
    raw = d.get("log_tail", "[]")
    try:
        d["log_tail"] = json.loads(raw)
    except Exception:
        d["log_tail"] = []
    return d


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
