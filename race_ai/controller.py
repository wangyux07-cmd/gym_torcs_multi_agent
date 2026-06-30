from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ControllerConfig:
    kp: float = 0.39
    kd: float = 0.19
    center_weight: float = 0.50
    steer_limit: float = 1.0
    straight_speed: float = 123.0
    corner_speed: float = 57.0
    accel_target_distance: float = 75.0
    sharp_turn_distance: float = 51.0
    side_turn_distance: float = 18.0
    brake_speed_margin: float = 5.0
    brake_amount: float = 0.29
    brake_track_distance: float = 32.0
    brake_min_speed: float = 55.0
    brake_target_speed: float = 140.0
    accel_up_step: float = 0.40
    accel_down_step: float = 0.20
    launch_speed: float = 10.0
    traction_slip_limit: float = 2.0
    traction_accel_reduction: float = 0.10
    gear_speeds: tuple[float, ...] = (0.0, 20.0, 40.0, 80.0, 100.0, 180.0)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ControllerConfig":
        data = dict(payload)
        if "gear_speeds" in data:
            data["gear_speeds"] = tuple(float(v) for v in data["gear_speeds"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gear_speeds"] = list(self.gear_speeds)
        return data


@dataclass
class DriverAction:
    steer: float
    accel: float
    brake: float
    gear: int


class RuleBasedDriver:
    """Deterministic TORCS controller intended as the stable product baseline."""

    def __init__(self, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self.prev_error = 0.0

    def reset(self) -> None:
        self.prev_error = 0.0

    def act(
        self,
        sensors: Mapping[str, Any],
        previous_accel: float = 0.0,
        target_speed_multiplier: float = 1.0,
    ) -> DriverAction:
        steer = self._steer(sensors)
        accel = self._throttle(sensors, steer, previous_accel, target_speed_multiplier=target_speed_multiplier)
        brake = self._brake(sensors)
        if brake > 0.0:
            accel = 0.0
        accel = self._traction_control(sensors, accel)
        gear = self._gear(sensors)
        return DriverAction(steer=steer, accel=accel, brake=brake, gear=gear)

    def act_with_steer(
        self,
        sensors: Mapping[str, Any],
        steer: float,
        previous_accel: float = 0.0,
        target_speed_multiplier: float = 1.0,
    ) -> DriverAction:
        accel = self._throttle(sensors, steer, previous_accel, target_speed_multiplier=target_speed_multiplier)
        brake = self._brake(sensors)
        if brake > 0.0:
            accel = 0.0
        accel = self._traction_control(sensors, accel)
        gear = self._gear(sensors)
        return DriverAction(steer=steer, accel=accel, brake=brake, gear=gear)

    def _steer(self, sensors: Mapping[str, Any]) -> float:
        cfg = self.config
        track = list(sensors.get("track", [200.0] * 19))
        track_pos = float(sensors.get("trackPos", 0.0))

        close_left = min(track[3:9])
        close_right = min(track[10:16])
        ahead = float(track[9])

        angles = [-10.0, -5.0, 0.0, 5.0, 10.0]
        center_distances = [float(track[i + 9]) for i in range(-2, 3)]
        total_distance = sum(center_distances)
        if total_distance > 0.0:
            angle_error = -sum(a * d for a, d in zip(angles, center_distances)) / total_distance / 10.0
        else:
            angle_error = 0.0

        if close_left < cfg.side_turn_distance or close_right < cfg.side_turn_distance or ahead < cfg.sharp_turn_distance:
            if close_left < close_right:
                error = -16.0 / max(close_left, 1.0)
            else:
                error = 16.0 / max(close_right, 1.0)
            return self._pid(error)

        if (total_distance / 5.0) > 80.0 or ahead > 110.0:
            return 0.0

        pos_error = -track_pos
        total_error = angle_error + pos_error * cfg.center_weight
        return self._pid(total_error)

    def _pid(self, error: float) -> float:
        cfg = self.config
        derivative = error - self.prev_error
        steer = cfg.kp * error + cfg.kd * derivative
        self.prev_error = error
        return clip(steer, -cfg.steer_limit, cfg.steer_limit)

    def _target_speed(self, sensors: Mapping[str, Any], target_speed_multiplier: float = 1.0) -> float:
        track = list(sensors.get("track", [200.0] * 19))
        ahead = float(track[9])
        if ahead < self.config.accel_target_distance:
            return self.config.corner_speed * target_speed_multiplier
        return self.config.straight_speed * target_speed_multiplier

    def _throttle(
        self,
        sensors: Mapping[str, Any],
        steer: float,
        previous_accel: float,
        target_speed_multiplier: float = 1.0,
    ) -> float:
        cfg = self.config
        speed_x = float(sensors.get("speedX", 0.0))
        target_speed = self._target_speed(sensors, target_speed_multiplier=target_speed_multiplier)
        if speed_x < target_speed - abs(steer) * 2.5:
            accel = previous_accel + cfg.accel_up_step
        else:
            accel = previous_accel - cfg.accel_down_step
        if speed_x < cfg.launch_speed:
            accel += 1.0 / (speed_x + 0.1)
        return clip(accel, 0.0, 1.0)

    def _brake(self, sensors: Mapping[str, Any]) -> float:
        cfg = self.config
        speed_x = float(sensors.get("speedX", 0.0))
        track = list(sensors.get("track", [200.0] * 19))
        center_distance = sum(float(track[i]) for i in (8, 9, 10)) / 3.0
        too_fast = speed_x - cfg.brake_speed_margin > cfg.brake_target_speed
        blind_corner = center_distance < cfg.brake_track_distance and speed_x > cfg.brake_min_speed
        return cfg.brake_amount if too_fast or blind_corner else 0.0

    def _traction_control(self, sensors: Mapping[str, Any], accel: float) -> float:
        wheels = list(sensors.get("wheelSpinVel", [0.0, 0.0, 0.0, 0.0]))
        slip = (float(wheels[2]) + float(wheels[3])) - (float(wheels[0]) + float(wheels[1]))
        if slip > self.config.traction_slip_limit:
            accel -= self.config.traction_accel_reduction
        return clip(accel, 0.0, 1.0)

    def _gear(self, sensors: Mapping[str, Any]) -> int:
        speed_x = float(sensors.get("speedX", 0.0))
        gear = 1
        for index, threshold in enumerate(self.config.gear_speeds):
            if speed_x > threshold:
                gear = index + 1
        return int(clip(gear, 1, 6))
