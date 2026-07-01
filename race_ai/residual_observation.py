from __future__ import annotations

import math

import numpy as np

from .controller import DriverAction

OBS_DIM = 35
DEFAULT_DELTA_SCALE = np.array([0.05, 0.08, 0.08], dtype=np.float32)  # steer, accel, brake


def build_observation(sensors: dict, rule_action: DriverAction) -> np.ndarray:
    track = list(sensors.get("track", [200.0] * 19))
    wheel_spin = list(sensors.get("wheelSpinVel", [0.0] * 4))
    dist_from_start = float(sensors.get("distFromStart", 0.0))
    progress_phase = 2.0 * math.pi * (dist_from_start % 3608.45) / 3608.45
    values = (
        [t / 200.0 for t in track]
        + [
            float(sensors.get("speedX", 0.0)) / 200.0,
            float(sensors.get("speedY", 0.0)) / 50.0,
            float(sensors.get("speedZ", 0.0)) / 50.0,
            float(sensors.get("trackPos", 0.0)),
            float(sensors.get("angle", 0.0)) / math.pi,
            float(np.sin(progress_phase)),
            float(np.cos(progress_phase)),
            float(sensors.get("damage", 0.0)) / 10000.0,
        ]
        + [w / 100.0 for w in wheel_spin]
        + [
            float(sensors.get("rpm", 0.0)) / 10000.0,
            rule_action.steer,
            rule_action.accel,
            rule_action.brake,
        ]
    )
    return np.asarray(values, dtype=np.float32)
