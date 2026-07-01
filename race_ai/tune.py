from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import optuna

from .config_io import load_controller_config

ROOT = Path(__file__).resolve().parent.parent
DRIVER_SCRIPT = ROOT / "race_rule_driver.py"

# (low, high) search ranges, centered loosely around the validated defaults
# in configs/rule_fast.json. Keep ranges tight enough that the controller
# stays recognizably "the same driver", just better tuned.
SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "kp": (0.20, 0.60),
    "kd": (0.05, 0.40),
    "center_weight": (0.20, 0.80),
    "straight_speed": (100.0, 200.0),
    "corner_speed": (40.0, 80.0),
    "accel_target_distance": (50.0, 110.0),
    "brake_amount": (0.10, 0.60),
    "brake_track_distance": (15.0, 50.0),
    "brake_min_speed": (35.0, 80.0),
    "accel_up_step": (0.20, 0.80),
    "accel_down_step": (0.05, 0.40),
}

# Trial fails to finish the lap: penalize, but reward whatever progress
# (distRaced) it made so the optimizer still gets a gradient toward "better".
DNF_PENALTY = 600.0
DNF_PROGRESS_COEF = 0.05
DAMAGE_COEF = 0.01
OFFTRACK_COEF = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-tune the rule-based TORCS controller with optuna.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--base-config", default="configs/rule_fast.json")
    parser.add_argument("--best-config-out", default="configs/rule_tuned.json")
    parser.add_argument("--runs-dir", default="runs/tune")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12000,
        help="Sim steps (~50/sec) before the in-process driver gives up and cleanly disconnects "
        "(sends meta=1). Keep --trial-timeout comfortably above this so the script's own "
        "graceful stop always fires first.",
    )
    parser.add_argument("--damage-limit", type=float, default=5000.0)
    parser.add_argument(
        "--trial-timeout",
        type=float,
        default=270.0,
        help="Hard kill safety net for a fully hung connection. Should stay above "
        "max_steps/50 + buffer - it should rarely if ever actually fire.",
    )
    parser.add_argument("--study-name", default="rule_controller_tuning")
    parser.add_argument("--storage", default=None, help="Optional optuna storage URL for resuming a study.")
    return parser.parse_args()


def sample_config(trial: optuna.Trial, base: dict[str, Any]) -> dict[str, Any]:
    config = dict(base)
    for name, (low, high) in SEARCH_SPACE.items():
        config[name] = trial.suggest_float(name, low, high)
    return config


def run_trial_episode(config: dict[str, Any], args: argparse.Namespace, trial_number: int) -> dict[str, str] | None:
    tmp_dir = ROOT / args.runs_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=str(tmp_dir)) as fh:
        json.dump(config, fh)
        config_path = Path(fh.name)

    cmd = [
        sys.executable,
        str(DRIVER_SCRIPT),
        "--host", args.host,
        "--port", str(args.port),
        "--config", str(config_path),
        "--run-name", f"tune_trial_{trial_number}",
        "--runs-dir", args.runs_dir,
        "--episodes", "1",
        "--target-laps", str(args.target_laps),
        "--max-steps", str(args.max_steps),
        "--damage-limit", str(args.damage_limit),
        "--stop-on-offtrack",
        "--print-every", "0",
    ]

    run_dir: Path | None = None
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=args.trial_timeout,
        )
        for line in result.stdout.splitlines():
            if line.startswith("run_dir="):
                run_dir = Path(line.split("=", 1)[1])
        if result.returncode != 0:
            print(f"trial {trial_number} subprocess failed:\n{result.stderr[-2000:]}")
    except subprocess.TimeoutExpired:
        print(f"trial {trial_number} timed out after {args.trial_timeout}s")
    finally:
        config_path.unlink(missing_ok=True)

    if run_dir is None:
        return None
    csv_path = run_dir / "episode_metrics.csv"
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else None


def score_episode(row: dict[str, str]) -> float:
    lap_complete = row.get("lap_complete") == "True"
    best_lap_time = float(row.get("best_lap_time") or 0.0)
    max_damage = float(row.get("max_damage") or 0.0)
    max_abs_track_pos = float(row.get("max_abs_track_pos") or 0.0)
    final_dist_raced = float(row.get("final_dist_raced") or 0.0)

    if lap_complete and best_lap_time > 0.0:
        return best_lap_time + DAMAGE_COEF * max_damage + OFFTRACK_COEF * max_abs_track_pos

    return DNF_PENALTY - DNF_PROGRESS_COEF * final_dist_raced


def make_objective(args: argparse.Namespace, base: dict[str, Any]):
    def objective(trial: optuna.Trial) -> float:
        config = sample_config(trial, base)
        row = run_trial_episode(config, args, trial.number)
        if row is None:
            return DNF_PENALTY
        score = score_episode(row)
        trial.set_user_attr("lap_complete", row.get("lap_complete"))
        trial.set_user_attr("best_lap_time", row.get("best_lap_time"))
        trial.set_user_attr("max_damage", row.get("max_damage"))
        print(
            f"trial {trial.number} score={score:.3f} lap_complete={row.get('lap_complete')} "
            f"best_lap={row.get('best_lap_time')} damage={row.get('max_damage')}"
        )
        return score

    return objective


def main() -> None:
    args = parse_args()
    base = load_controller_config(args.base_config).to_dict()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=args.storage is not None,
    )
    study.optimize(make_objective(args, base), n_trials=args.trials)

    best_config = dict(base)
    best_config.update(study.best_params)
    out_path = Path(args.best_config_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(best_config, fh, indent=2, sort_keys=True)

    print(f"best_score={study.best_value:.3f}")
    print(f"best_params={study.best_params}")
    print(f"Saved best config to {out_path}")


if __name__ == "__main__":
    main()
