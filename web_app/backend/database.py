from __future__ import annotations

import sqlite3
from pathlib import Path


def get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id  TEXT NOT NULL PRIMARY KEY,
                username TEXT NOT NULL DEFAULT 'Anonymous'
            );

            CREATE TABLE IF NOT EXISTS settings (
                user_id TEXT NOT NULL,
                key     TEXT NOT NULL,
                value   TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, key)
            );

            CREATE TABLE IF NOT EXISTS results (
                id           TEXT NOT NULL,
                user_id      TEXT NOT NULL,
                username     TEXT NOT NULL DEFAULT 'Anonymous',
                started_at   TEXT NOT NULL DEFAULT '',
                finished_at  TEXT NOT NULL DEFAULT '',
                status       TEXT NOT NULL DEFAULT '',
                reason       TEXT NOT NULL DEFAULT '',
                lap_time     REAL,
                frame_count  INTEGER,
                distance     REAL NOT NULL DEFAULT 0,
                total_reward REAL,
                best_model   TEXT NOT NULL DEFAULT '',
                log_tail     TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (id, user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_results_user
                ON results (user_id, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_results_lap
                ON results (status, lap_time ASC);
        """)
        # Migrate existing results table that may not have the username column
        try:
            conn.execute("ALTER TABLE results ADD COLUMN username TEXT NOT NULL DEFAULT 'Anonymous'")
        except Exception:
            pass
