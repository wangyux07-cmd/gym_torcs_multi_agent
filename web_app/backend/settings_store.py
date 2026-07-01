from __future__ import annotations

from pathlib import Path
from typing import Any

from .database import get_conn


DEFAULT_SETTINGS = {
    "torcs_path": "",
    "torcs_args": "-nofuel -nodamage -nolaptime",
    "window_title": "TORCS",
    "track_category": "road",
    "track_name": "corkscrew",
    "laps": "1",
    "driver_config": "configs/rule_fast.json",
}

# Settings that are machine-level: fall back to "default" user if the
# requesting user hasn't configured them yet.
_MACHINE_KEYS = {"torcs_path", "torcs_args", "window_title"}


class SettingsStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def load(self, user_id: str = "default") -> dict[str, Any]:
        settings = dict(DEFAULT_SETTINGS)
        try:
            with get_conn(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT key, value FROM settings WHERE user_id = ?", (user_id,)
                ).fetchall()
            for row in rows:
                if row["key"] in settings:
                    settings[row["key"]] = row["value"]

            # Fall back to "default" user for machine-level keys when empty
            if user_id != "default":
                missing = [k for k in _MACHINE_KEYS if not settings.get(k)]
                if missing:
                    placeholders = ",".join("?" * len(missing))
                    with get_conn(self.db_path) as conn:
                        fb_rows = conn.execute(
                            f"SELECT key, value FROM settings WHERE user_id = 'default' AND key IN ({placeholders})",
                            missing,
                        ).fetchall()
                    for row in fb_rows:
                        settings[row["key"]] = row["value"]
        except Exception:
            pass
        return settings

    def save(self, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self.load(user_id)
        for key in DEFAULT_SETTINGS:
            if key in payload:
                settings[key] = str(payload[key]).strip()
        try:
            with get_conn(self.db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO settings (user_id, key, value) VALUES (?, ?, ?)",
                    [(user_id, k, v) for k, v in settings.items()],
                )
        except Exception:
            pass
        return settings
