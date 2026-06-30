from __future__ import annotations

import math
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import snakeoil3_gym as snakeoil3

from .controller import ControllerConfig, DriverAction, RuleBasedDriver, clip
from .sector_reference import SectorReference
from .speed_profile import SpeedProfile


SECTOR_OBS_DIM = 39


class SectorResidualEnv(gym.Env):
    """Sector-level residual RL for TORCS.

    The rule controller and speed profile remain responsible for stable driving.
    SAC acts once per track sector and can only request a small speed change by
    default.
    This makes the credit assignment line up with lap-time improvement instead
    of asking the policy to discover low-level driving from scratch.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3001,
        config: ControllerConfig | None = None,
        profile: SpeedProfile | None = None,
        reference: SectorReference | None = None,
        target_laps: int = 1,
        max_frames: int = 12000,
        damage_limit: float = 5000.0,
        stop_on_offtrack: bool = True,
        stuck_frames_limit: int = 300,
        backward_angle_limit: float = 1.25,
        backward_speed_limit: float = -1.0,
        restart_wait_sec: float = 1.0,
        speed_delta: float = 0.04,
        brake_delta_kmh: float = 5.0,
        enable_brake_delta: bool = False,
        locked_sectors: tuple[int, ...] = tuple(range(44, 51)) + tuple(range(61, 67)),
        min_speed_multiplier: float = 0.92,
        max_speed_multiplier: float = 1.44,
        min_brake_target_speed: float = 120.0,
        max_brake_target_speed: float = 170.0,
        sector_step_reward: float = 8.0,
        progress_reward: float = 0.0,
        finish_bonus: float = 300.0,
        finish_step_reward: float = 0.08,
        failure_penalty: float = 240.0,
        offtrack_speed_penalty_scale: float = 0.9,
        backward_speed_penalty_scale: float = 0.6,
        edge_penalty_scale: float = 25.0,
        heading_penalty_scale: float = 10.0,
        speed_edge_penalty_scale: float = 8.0,
        posture_penalty_scale: float = 18.0,
        speed_posture_penalty_scale: float = 10.0,
        unsafe_reward_gate: float = 0.55,
        damage_penalty_scale: float = 1.0,
        action_penalty_scale: float = 0.03,
        action_change_penalty_scale: float = 0.12,
    ):
        super().__init__()
        if profile is None:
            raise ValueError("profile is required")
        if reference is None:
            raise ValueError("reference is required")
        self.host = host
        self.port = port
        self.config = config or ControllerConfig()
        self.profile = profile
        self.reference = reference
        self.target_laps = target_laps
        self.max_frames = max_frames
        self.damage_limit = damage_limit
        self.stop_on_offtrack = stop_on_offtrack
        self.stuck_frames_limit = stuck_frames_limit
        self.backward_angle_limit = backward_angle_limit
        self.backward_speed_limit = backward_speed_limit
        self.restart_wait_sec = restart_wait_sec
        self.speed_delta = speed_delta
        self.brake_delta_kmh = brake_delta_kmh
        self.enable_brake_delta = enable_brake_delta
        self.locked_sectors = frozenset(int(value) for value in locked_sectors)
        self.min_speed_multiplier = min_speed_multiplier
        self.max_speed_multiplier = max_speed_multiplier
        self.min_brake_target_speed = min_brake_target_speed
        self.max_brake_target_speed = max_brake_target_speed
        self.sector_step_reward = sector_step_reward
        self.progress_reward = progress_reward
        self.finish_bonus = finish_bonus
        self.finish_step_reward = finish_step_reward
        self.failure_penalty = failure_penalty
        self.offtrack_speed_penalty_scale = offtrack_speed_penalty_scale
        self.backward_speed_penalty_scale = backward_speed_penalty_scale
        self.edge_penalty_scale = edge_penalty_scale
        self.heading_penalty_scale = heading_penalty_scale
        self.speed_edge_penalty_scale = speed_edge_penalty_scale
        self.posture_penalty_scale = posture_penalty_scale
        self.speed_posture_penalty_scale = speed_posture_penalty_scale
        self.unsafe_reward_gate = unsafe_reward_gate
        self.damage_penalty_scale = damage_penalty_scale
        self.action_penalty_scale = action_penalty_scale
        self.action_change_penalty_scale = action_change_penalty_scale

        self.driver = RuleBasedDriver(self.config)
        self.client: snakeoil3.Client | None = None
        self.sensors: dict[str, Any] = {}

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(SECTOR_OBS_DIM,), dtype=np.float32)
        action_dim = 2 if self.enable_brake_delta else 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def _distance(self, sensors: dict[str, Any]) -> float:
        return float(sensors.get("distRaced", 0.0))

    def _profile_distance(self, sensors: dict[str, Any]) -> float:
        return float(sensors.get("distFromStart", sensors.get("distRaced", 0.0)))

    def _sector_entry_cap(self, sector_index: int) -> float | None:
        # Sector 49 failures come from entering the downhill corner too hot.
        # Keep the approach stable, then let SAC search for gains elsewhere.
        if sector_index in (44, 45):
            return 1.05
        if sector_index in (46, 47, 48, 49, 50):
            return 1.0
        if sector_index in (61, 62):
            return 1.05
        if sector_index in (63, 64, 65, 66):
            return 1.0
        return None

    def _sector_speed_floor(self, sector_index: int) -> float | None:
        # Safe straight/exit sectors from telemetry: low edge risk, low heading,
        # and no recent unsafe posture. Floors only apply below the safety cap.
        floors = {
            13: 1.10,
            24: 1.22,
            25: 1.22,
            40: 1.10,
            41: 1.30,
            42: 1.30,
            51: 1.08,
            56: 1.18,
            67: 1.40,
            70: 1.40,
        }
        return floors.get(sector_index)

    def _safety_cap(self, sensors: dict[str, Any], rule_steer: float) -> float:
        track = list(sensors.get("track", [200.0] * 19))
        ahead = float(track[9])
        side_clearance = min(float(value) for value in track[4:15])
        angle_abs = abs(float(sensors.get("angle", 0.0)))
        track_pos_abs = abs(float(sensors.get("trackPos", 0.0)))
        steer_abs = abs(rule_steer)
        sector = self.reference.sector_for_distance(self._distance(sensors))

        caution_cap = max(self.min_speed_multiplier, min(1.0, self.profile.caution_multiplier))
        sector_cap = self._sector_entry_cap(sector.index)
        unsafe_posture = track_pos_abs > 0.88 and angle_abs > 0.22
        severe_posture = track_pos_abs > 0.94 and angle_abs > 0.16
        if angle_abs > 0.55 or severe_posture:
            return caution_cap
        if ahead < 45.0 or side_clearance < 14.0 or steer_abs > 0.28 or angle_abs > 0.36 or unsafe_posture:
            return min(1.0, sector_cap) if sector_cap is not None else 1.0
        if ahead < 75.0 or side_clearance < 24.0 or steer_abs > 0.18 or angle_abs > 0.24:
            return min(1.04, sector_cap) if sector_cap is not None else 1.04
        return min(self.max_speed_multiplier, sector_cap) if sector_cap is not None else self.max_speed_multiplier

    def _action_values(self, sensors: dict[str, Any], action: np.ndarray) -> tuple[float, float]:
        sector = self.reference.sector_for_distance(self._distance(sensors))
        speed_action = float(action[0])
        if sector.index in self.locked_sectors:
            speed_action = 0.0
        brake_action = float(action[1]) if self.enable_brake_delta and len(action) > 1 else 0.0
        return speed_action, brake_action

    def _decode_action(
        self,
        sensors: dict[str, Any],
        rule_action: DriverAction,
        action: np.ndarray,
    ) -> tuple[float, float, float, float]:
        profile_distance = self._profile_distance(sensors)
        speed_action, brake_action = self._action_values(sensors, action)
        base_multiplier = self.profile.raw_multiplier(profile_distance)
        desired_multiplier = base_multiplier + speed_action * self.speed_delta
        desired_multiplier = clip(desired_multiplier, self.min_speed_multiplier, self.max_speed_multiplier)
        sector = self.reference.sector_for_distance(self._distance(sensors))
        safety_cap = self._safety_cap(sensors, rule_action.steer)
        desired_multiplier = min(desired_multiplier, safety_cap)
        speed_floor = self._sector_speed_floor(sector.index)
        if speed_floor is not None:
            desired_multiplier = max(desired_multiplier, min(speed_floor, safety_cap))

        base_brake_speed = self.config.brake_target_speed * self.profile.brake_multiplier(profile_distance)
        desired_brake_speed = base_brake_speed + brake_action * self.brake_delta_kmh
        desired_brake_speed = clip(desired_brake_speed, self.min_brake_target_speed, self.max_brake_target_speed)
        return desired_multiplier, desired_brake_speed, speed_action, brake_action

    def _observation(self) -> np.ndarray:
        sensors = self.sensors
        track = list(sensors.get("track", [200.0] * 19))
        wheel_spin = list(sensors.get("wheelSpinVel", [0.0] * 4))
        distance = self._distance(sensors)
        phase = 2.0 * math.pi * (distance % self.reference.lap_length) / self.reference.lap_length
        sector = self.reference.sector_for_distance(distance)
        sector_width = max(1e-6, sector.end - sector.start)
        sector_progress = clip((distance - sector.start) / sector_width, 0.0, 1.0)
        profile_distance = self._profile_distance(sensors)
        base_multiplier = self.profile.raw_multiplier(profile_distance)
        base_brake_speed = self.config.brake_target_speed * self.profile.brake_multiplier(profile_distance)

        values = (
            [float(value) / 200.0 for value in track]
            + [
                float(sensors.get("speedX", 0.0)) / 220.0,
                float(sensors.get("speedY", 0.0)) / 50.0,
                float(sensors.get("speedZ", 0.0)) / 50.0,
                float(sensors.get("trackPos", 0.0)),
                float(sensors.get("angle", 0.0)) / math.pi,
                math.sin(phase),
                math.cos(phase),
                sector.index / max(1, self.reference.sector_count - 1),
                sector_progress,
                sector.reference_steps / 200.0,
                sector.reference_speed / 200.0,
                base_multiplier,
                base_brake_speed / 200.0,
                float(sensors.get("damage", 0.0)) / 10000.0,
            ]
            + [float(value) / 100.0 for value in wheel_spin]
            + [
                getattr(self, "previous_speed_delta", 0.0),
                getattr(self, "previous_brake_delta", 0.0),
            ]
        )
        return np.asarray(values, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.client is not None:
            self.client.R.d["meta"] = 1
            self.client.respond_to_server()
            self.client.shutdown()
            self.client = None
            time.sleep(self.restart_wait_sec)
        self.client = snakeoil3.Client(H=self.host, p=self.port)
        self.client.get_servers_input()
        self.sensors = self.client.S.d

        self.driver.reset()
        self.completed_laps = 0
        self.previous_lap_time = 0.0
        self.frame_count = 0
        self.best_progress = float(self.sensors.get("distRaced", 0.0))
        self.best_progress_frame = 0
        self.previous_damage = float(self.sensors.get("damage", 0.0))
        self.previous_accel = 0.0
        self.previous_speed_delta = 0.0
        self.previous_brake_delta = 0.0
        return self._observation(), {}

    def _termination_reason(self, sensors: dict[str, Any]) -> str | None:
        if self.completed_laps >= self.target_laps:
            return "target_laps"
        if float(sensors.get("damage", 0.0)) >= self.damage_limit:
            return "damage_limit"
        speed_x = float(sensors.get("speedX", 0.0))
        angle = float(sensors.get("angle", 0.0))
        if abs(angle) > self.backward_angle_limit or speed_x < self.backward_speed_limit:
            return "backward"
        if self.stop_on_offtrack and abs(float(sensors.get("trackPos", 0.0))) > 1.0:
            return "offtrack"
        if (self.frame_count - self.best_progress_frame) >= self.stuck_frames_limit:
            return "stuck"
        return None

    def step(self, action: np.ndarray):
        assert self.client is not None, "call reset() before step()"
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        start_distance = self._distance(self.sensors)
        sector = self.reference.sector_for_distance(start_distance)
        target_distance = sector.end
        start_frame = self.frame_count
        start_progress = float(self.sensors.get("distRaced", 0.0))
        start_damage = float(self.sensors.get("damage", 0.0))
        max_abs_track_pos = 0.0
        max_abs_angle = 0.0
        edge_risk_sum = 0.0
        heading_risk_sum = 0.0
        speed_edge_risk_sum = 0.0
        posture_risk_sum = 0.0
        speed_posture_risk_sum = 0.0
        unsafe_frames = 0
        max_speed_x = 0.0
        start_profile_distance = self._profile_distance(self.sensors)
        speed_multiplier = self.profile.raw_multiplier(start_profile_distance)
        brake_target_speed = self.config.brake_target_speed * self.profile.brake_multiplier(start_profile_distance)
        speed_action = 0.0
        brake_action = 0.0
        sector_entry_cap = self._sector_entry_cap(sector.index)
        sector_speed_floor = self._sector_speed_floor(sector.index)
        reason: str | None = None

        while True:
            sensors = self.sensors
            base_action = self.driver.act(sensors, previous_accel=self.previous_accel)
            speed_multiplier, brake_target_speed, speed_action, brake_action = self._decode_action(sensors, base_action, action)
            driver_action = self.driver.act_with_steer(
                sensors,
                steer=base_action.steer,
                previous_accel=self.previous_accel,
                target_speed_multiplier=speed_multiplier,
                brake_target_speed=brake_target_speed,
            )

            self.client.R.d["steer"] = driver_action.steer
            self.client.R.d["accel"] = driver_action.accel
            self.client.R.d["brake"] = driver_action.brake
            self.client.R.d["gear"] = driver_action.gear
            self.client.R.d["meta"] = 0
            self.previous_accel = driver_action.accel
            self.client.respond_to_server()

            self.client.get_servers_input()
            self.sensors = self.client.S.d
            self.frame_count += 1

            lap_time = float(self.sensors.get("lastLapTime", 0.0))
            if lap_time > 0.0 and lap_time != self.previous_lap_time:
                self.completed_laps += 1
                self.previous_lap_time = lap_time

            dist_raced = float(self.sensors.get("distRaced", 0.0))
            if dist_raced > self.best_progress:
                self.best_progress = dist_raced
                self.best_progress_frame = self.frame_count
            speed_x = max(0.0, float(self.sensors.get("speedX", 0.0)))
            track_pos_abs = abs(float(self.sensors.get("trackPos", 0.0)))
            angle_abs = abs(float(self.sensors.get("angle", 0.0)))
            max_speed_x = max(max_speed_x, speed_x)
            max_abs_track_pos = max(max_abs_track_pos, track_pos_abs)
            max_abs_angle = max(max_abs_angle, angle_abs)
            edge_excess = max(0.0, track_pos_abs - 0.80)
            heading_excess = max(0.0, angle_abs - 0.25)
            posture_edge = max(0.0, track_pos_abs - 0.86)
            posture_angle = max(0.0, angle_abs - 0.16)
            posture_risk = posture_edge * posture_angle
            edge_risk_sum += edge_excess * edge_excess
            heading_risk_sum += heading_excess * heading_excess
            speed_edge_risk_sum += (speed_x / 100.0) * edge_excess * edge_excess
            posture_risk_sum += posture_risk
            speed_posture_risk_sum += (speed_x / 100.0) * posture_risk
            if track_pos_abs > 0.90 and angle_abs > 0.20:
                unsafe_frames += 1

            reason = self._termination_reason(self.sensors)
            reached_sector = self._distance(self.sensors) >= target_distance or dist_raced >= target_distance
            if reason or reached_sector or self.frame_count >= self.max_frames:
                break

        sector_frames = max(1, self.frame_count - start_frame)
        progress = float(self.sensors.get("distRaced", 0.0)) - start_progress
        damage_delta = max(0.0, float(self.sensors.get("damage", 0.0)) - start_damage)
        speed_delta = speed_action
        brake_delta = brake_action
        sector_time_delta = (sector.reference_steps - sector_frames) / max(1.0, float(sector.reference_steps))
        sector_time_delta = clip(sector_time_delta, -1.0, 1.0)
        edge_risk = edge_risk_sum / sector_frames
        heading_risk = heading_risk_sum / sector_frames
        speed_edge_risk = speed_edge_risk_sum / sector_frames
        posture_risk = posture_risk_sum / sector_frames
        speed_posture_risk = speed_posture_risk_sum / sector_frames
        unsafe_fraction = unsafe_frames / sector_frames
        safety_gate = clip(1.0 - unsafe_fraction / max(1e-6, self.unsafe_reward_gate), 0.0, 1.0)
        if sector_time_delta > 0.0:
            reward_sector_time = self.sector_step_reward * sector_time_delta * safety_gate
        else:
            reward_sector_time = self.sector_step_reward * sector_time_delta
        reward_progress = self.progress_reward * progress
        penalty_edge = self.edge_penalty_scale * edge_risk
        penalty_heading = self.heading_penalty_scale * heading_risk
        penalty_speed_edge = self.speed_edge_penalty_scale * speed_edge_risk
        penalty_posture = self.posture_penalty_scale * posture_risk
        penalty_speed_posture = self.speed_posture_penalty_scale * speed_posture_risk

        reward = (
            reward_sector_time
            + reward_progress
            - penalty_edge
            - penalty_heading
            - penalty_speed_edge
            - penalty_posture
            - penalty_speed_posture
            - self.damage_penalty_scale * damage_delta
            - self.action_penalty_scale * (speed_delta * speed_delta + 0.4 * brake_delta * brake_delta)
            - self.action_change_penalty_scale
            * (abs(speed_delta - self.previous_speed_delta) + 0.4 * abs(brake_delta - self.previous_brake_delta))
        )
        self.previous_speed_delta = speed_delta
        self.previous_brake_delta = brake_delta

        terminated = False
        truncated = False
        if self.completed_laps >= self.target_laps:
            terminated = True
            reason = "target_laps"
            reward += self.finish_bonus
            reward += self.finish_step_reward * max(0, self.reference.reference_lap_steps - self.frame_count)
        elif reason is not None:
            terminated = True
            remaining = max(0.0, self.reference.lap_length - float(self.sensors.get("distRaced", 0.0)))
            final_speed = max(0.0, float(self.sensors.get("speedX", 0.0)))
            final_track_pos_abs = abs(float(self.sensors.get("trackPos", 0.0)))
            final_angle_abs = abs(float(self.sensors.get("angle", 0.0)))
            speed_penalty = 0.0
            if reason == "offtrack":
                speed_penalty = self.offtrack_speed_penalty_scale * final_speed
            elif reason == "backward":
                speed_penalty = self.backward_speed_penalty_scale * max(final_speed, max_speed_x)
            state_penalty = 35.0 * max(0.0, final_track_pos_abs - 0.75) + 20.0 * max(0.0, final_angle_abs - 0.6)
            reward -= self.failure_penalty + 0.03 * remaining + speed_penalty + state_penalty
        elif self.frame_count >= self.max_frames:
            truncated = True
            reason = "max_frames"

        if terminated or truncated:
            self.client.R.d["meta"] = 1
            self.client.respond_to_server()

        info = {
            "reason": reason,
            "sector": sector.index,
            "sector_steps": sector_frames,
            "reference_sector_steps": sector.reference_steps,
            "frame_count": self.frame_count,
            "dist_raced": float(self.sensors.get("distRaced", 0.0)),
            "lap_time": self.previous_lap_time,
            "speed_x": float(self.sensors.get("speedX", 0.0)),
            "track_pos": float(self.sensors.get("trackPos", 0.0)),
            "angle": float(self.sensors.get("angle", 0.0)),
            "speed_multiplier": float(speed_multiplier),
            "sector_entry_cap": "" if sector_entry_cap is None else float(sector_entry_cap),
            "sector_speed_floor": "" if sector_speed_floor is None else float(sector_speed_floor),
            "brake_target_speed": float(brake_target_speed),
            "speed_delta_action": speed_delta,
            "brake_delta_action": brake_delta,
            "max_abs_track_pos": max_abs_track_pos,
            "max_abs_angle": max_abs_angle,
            "max_speed_x": max_speed_x,
            "edge_risk": edge_risk,
            "heading_risk": heading_risk,
            "speed_edge_risk": speed_edge_risk,
            "posture_risk": posture_risk,
            "speed_posture_risk": speed_posture_risk,
            "unsafe_fraction": unsafe_fraction,
            "safety_gate": safety_gate,
            "sector_time_delta": sector_time_delta,
            "reward_sector_time": reward_sector_time,
            "reward_progress": reward_progress,
            "penalty_edge": penalty_edge,
            "penalty_heading": penalty_heading,
            "penalty_speed_edge": penalty_speed_edge,
            "penalty_posture": penalty_posture,
            "penalty_speed_posture": penalty_speed_posture,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.client is not None:
            self.client.R.d["meta"] = 1
            self.client.respond_to_server()
            self.client.shutdown()
            self.client = None
