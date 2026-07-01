from __future__ import annotations

import math
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import snakeoil3_gym as snakeoil3

from .controller import ControllerConfig, DriverAction, RuleBasedDriver, clip
from .residual_observation import DEFAULT_DELTA_SCALE, OBS_DIM, build_observation


class ResidualDriverEnv(gym.Env):
    """RL learns a small correction on top of RuleBasedDriver's action.

    One TORCS client connection is reused across episodes: each episode end
    sends (meta 1) so the practice race (restart="yes") restarts in place,
    and the next reset() opens a fresh Client to re-handshake - cheap and
    avoids the OS-level process kills that make TORCS exit uncleanly.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3001,
        config: ControllerConfig | None = None,
        target_laps: int = 1,
        max_steps: int = 12000,
        damage_limit: float = 5000.0,
        stop_on_offtrack: bool = True,
        stuck_steps_limit: int = 300,
        backward_angle_limit: float = 1.25,
        backward_speed_limit: float = -1.0,
        restart_wait_sec: float = 1.0,
        action_mode: str = "speed_target",
        delta_scale: tuple[float, float, float] | np.ndarray = tuple(DEFAULT_DELTA_SCALE.tolist()),
        min_speed_multiplier: float = 0.90,
        max_speed_multiplier: float = 1.08,
        multiplier_step_limit: float = 0.01,
        caution_speed_multiplier: float = 0.98,
        boost_start_distance: float = 650.0,
        boost_deadband: float = 0.25,
        lap_bonus: float = 250.0,
        offtrack_penalty: float = 150.0,
        damage_penalty_scale: float = 1.0,
        residual_penalty_scale: float = 0.02,
        heading_penalty_scale: float = 0.35,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.config = config or ControllerConfig()
        self.target_laps = target_laps
        self.max_steps = max_steps
        self.damage_limit = damage_limit
        self.stop_on_offtrack = stop_on_offtrack
        self.stuck_steps_limit = stuck_steps_limit
        self.backward_angle_limit = backward_angle_limit
        self.backward_speed_limit = backward_speed_limit
        self.restart_wait_sec = restart_wait_sec
        if action_mode not in {"speed_target", "slow_only", "speed_only", "full"}:
            raise ValueError("action_mode must be 'speed_target', 'slow_only', 'speed_only', or 'full'")
        self.action_mode = action_mode
        self.delta_scale = np.asarray(delta_scale, dtype=np.float32)
        self.min_speed_multiplier = min_speed_multiplier
        self.max_speed_multiplier = max_speed_multiplier
        self.multiplier_step_limit = multiplier_step_limit
        self.caution_speed_multiplier = caution_speed_multiplier
        self.boost_start_distance = boost_start_distance
        self.boost_deadband = boost_deadband
        self.lap_bonus = lap_bonus
        self.offtrack_penalty = offtrack_penalty
        self.damage_penalty_scale = damage_penalty_scale
        self.residual_penalty_scale = residual_penalty_scale
        self.heading_penalty_scale = heading_penalty_scale

        self.driver = RuleBasedDriver(self.config)
        self.client: snakeoil3.Client | None = None
        self.sensors: dict[str, Any] = {}
        self.pending_rule_action: DriverAction | None = None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        action_dim = 1 if self.action_mode == "speed_target" else 2 if self.action_mode in {"slow_only", "speed_only"} else 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def _safe_speed_multiplier(self, sensors: dict[str, Any], raw_action: float) -> float:
        dist = float(sensors.get("distRaced", sensors.get("distFromStart", 0.0)))
        if dist < self.boost_start_distance:
            desired = 1.0
        else:
            boost_signal = max(0.0, float(raw_action) - self.boost_deadband) / max(1e-6, 1.0 - self.boost_deadband)
            desired = 1.0 + boost_signal * (self.max_speed_multiplier - 1.0)
        desired = clip(desired, self.min_speed_multiplier, self.max_speed_multiplier)

        track = list(sensors.get("track", [200.0] * 19))
        ahead = float(track[9])
        side_clearance = min(float(v) for v in track[4:15])
        angle_abs = abs(float(sensors.get("angle", 0.0)))
        track_pos_abs = abs(float(sensors.get("trackPos", 0.0)))
        steer_abs = abs(self.pending_rule_action.steer) if self.pending_rule_action is not None else 0.0

        safety_cap = self.max_speed_multiplier
        caution_cap = max(self.min_speed_multiplier, min(1.0, self.caution_speed_multiplier))
        if angle_abs > 0.55 or track_pos_abs > 0.72:
            safety_cap = caution_cap
        elif ahead < 45.0 or side_clearance < 14.0 or steer_abs > 0.28 or angle_abs > 0.36 or track_pos_abs > 0.55:
            safety_cap = 1.0
        elif ahead < 75.0 or side_clearance < 24.0 or steer_abs > 0.18 or angle_abs > 0.24 or track_pos_abs > 0.38:
            safety_cap = min(safety_cap, 1.04)

        desired = min(desired, safety_cap)
        previous = getattr(self, "last_speed_multiplier", 1.0)
        if desired < previous:
            return clip(desired, self.min_speed_multiplier, previous)
        upper = min(self.max_speed_multiplier, previous + self.multiplier_step_limit)
        return clip(desired, previous, upper)

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
        self.prev_dist = float(self.sensors.get("distRaced", 0.0))
        self.best_progress = self.prev_dist
        self.best_progress_step = 0
        self.step_count = 0
        self.last_accel = 0.0
        self.last_speed_multiplier = 1.0
        self.previous_speed_multiplier = 1.0
        self.sum_speed_multiplier = 0.0
        self.max_speed_multiplier_seen = 1.0
        self.previous_damage = float(self.sensors.get("damage", 0.0))

        self.pending_rule_action = self.driver.act(self.sensors, previous_accel=0.0, target_speed_multiplier=1.0)
        obs = build_observation(self.sensors, self.pending_rule_action)
        return obs, {}

    def step(self, action: np.ndarray):
        assert self.client is not None, "call reset() before step()"
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.action_mode == "speed_target":
            base_rule_action = self.pending_rule_action
            assert base_rule_action is not None
            speed_multiplier = self._safe_speed_multiplier(self.sensors, float(action[0]))
            self.last_speed_multiplier = speed_multiplier
            rule_action = self.driver.act_with_steer(
                self.sensors,
                steer=base_rule_action.steer,
                previous_accel=self.last_accel,
                target_speed_multiplier=speed_multiplier,
            )
            delta = np.zeros(3, dtype=np.float32)
        else:
            rule_action = self.pending_rule_action
            assert rule_action is not None
            speed_multiplier = 1.0
        self.sum_speed_multiplier += speed_multiplier
        self.max_speed_multiplier_seen = max(self.max_speed_multiplier_seen, speed_multiplier)
        if self.action_mode == "speed_target":
            pass
        elif self.action_mode == "slow_only":
            delta = np.array(
                [
                    0.0,
                    min(0.0, float(action[0])) * self.delta_scale[1],
                    max(0.0, float(action[1])) * self.delta_scale[2],
                ],
                dtype=np.float32,
            )
        elif self.action_mode == "speed_only":
            delta = np.array([0.0, action[0] * self.delta_scale[1], action[1] * self.delta_scale[2]], dtype=np.float32)
        else:
            delta = action * self.delta_scale

        final_steer = clip(rule_action.steer + float(delta[0]), -1.0, 1.0)
        final_accel = clip(rule_action.accel + float(delta[1]), 0.0, 1.0)
        final_brake = clip(rule_action.brake + float(delta[2]), 0.0, 1.0)
        if final_brake > 0.0:
            final_accel = 0.0
        self.last_accel = final_accel

        self.client.R.d["steer"] = final_steer
        self.client.R.d["accel"] = final_accel
        self.client.R.d["brake"] = final_brake
        self.client.R.d["gear"] = rule_action.gear
        self.client.R.d["meta"] = 0
        self.client.respond_to_server()

        self.client.get_servers_input()
        sensors = self.client.S.d
        self.sensors = sensors

        lap_time = float(sensors.get("lastLapTime", 0.0))
        if lap_time > 0.0 and lap_time != self.previous_lap_time:
            self.completed_laps += 1
            self.previous_lap_time = lap_time

        dist_raced = float(sensors.get("distRaced", 0.0))
        progress = dist_raced - self.prev_dist
        self.prev_dist = dist_raced
        if dist_raced > self.best_progress:
            self.best_progress = dist_raced
            self.best_progress_step = self.step_count
        stuck = (self.step_count - self.best_progress_step) >= self.stuck_steps_limit

        track = list(sensors.get("track", [200.0] * 19))
        track_pos = float(sensors.get("trackPos", 0.0))
        angle = float(sensors.get("angle", 0.0))
        damage = float(sensors.get("damage", 0.0))
        damage_delta = max(0.0, damage - self.previous_damage)
        self.previous_damage = damage

        speed_along_track = float(sensors.get("speedX", 0.0)) * max(0.0, math.cos(angle))
        speed_x = float(sensors.get("speedX", 0.0))
        center_gate = max(0.0, 1.0 - abs(track_pos))
        heading_gate = max(0.0, math.cos(angle))
        reward = (
            0.25 * progress * center_gate * heading_gate
            + 0.01 * speed_along_track * center_gate
            - 0.08 * abs(track_pos)
            - 0.03 * abs(angle)
            - self.damage_penalty_scale * damage_delta
            - self.residual_penalty_scale * float(np.sum(np.square(delta / np.maximum(self.delta_scale, 1e-6))))
        )
        heading_excess = max(0.0, abs(angle) - 0.22)
        edge_excess = max(0.0, abs(track_pos) - 0.45)
        reward -= self.heading_penalty_scale * heading_excess * heading_excess
        reward -= 0.25 * edge_excess * edge_excess
        if speed_x > 20.0:
            reward -= 0.004 * speed_x * heading_excess
        if self.action_mode == "speed_target":
            reward += 0.02 * speed_along_track * center_gate * max(0.0, speed_multiplier - 1.0)
            reward -= 0.03 * abs(speed_multiplier - getattr(self, "previous_speed_multiplier", speed_multiplier))
            self.previous_speed_multiplier = speed_multiplier

        terminated = False
        truncated = False
        reason = None
        if self.completed_laps >= self.target_laps:
            terminated, reason = True, "target_laps"
            reward += self.lap_bonus
        elif damage >= self.damage_limit:
            terminated, reason = True, "damage_limit"
            reward -= 30.0
        elif abs(angle) > self.backward_angle_limit or speed_x < self.backward_speed_limit:
            terminated, reason = True, "backward"
            reward -= self.offtrack_penalty
        elif self.stop_on_offtrack and abs(track_pos) > 1.0:
            terminated, reason = True, "offtrack"
            reward -= self.offtrack_penalty
        elif stuck:
            terminated, reason = True, "stuck"
            reward -= 30.0
        elif self.step_count >= self.max_steps:
            truncated, reason = True, "max_steps"

        if terminated or truncated:
            self.client.R.d["meta"] = 1
            self.client.respond_to_server()
            self.pending_rule_action = rule_action
        else:
            self.pending_rule_action = self.driver.act(
                sensors,
                previous_accel=self.last_accel,
                target_speed_multiplier=getattr(self, "last_speed_multiplier", 1.0),
            )

        obs = build_observation(sensors, self.pending_rule_action)
        info = {
            "reason": reason,
            "dist_raced": dist_raced,
            "best_progress": self.best_progress,
            "lap_time": self.previous_lap_time,
            "track_pos": track_pos,
            "angle": angle,
            "speed_x": speed_x,
            "progress": progress,
            "rule_steer": rule_action.steer,
            "rule_accel": rule_action.accel,
            "rule_brake": rule_action.brake,
            "delta_steer": float(delta[0]),
            "delta_accel": float(delta[1]),
            "delta_brake": float(delta[2]),
            "speed_multiplier": float(speed_multiplier),
            "mean_speed_multiplier": float(self.sum_speed_multiplier / max(1, self.step_count)),
            "max_speed_multiplier": float(self.max_speed_multiplier_seen),
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.client is not None:
            # send meta=1 in case training stopped mid-episode before step() sent it
            self.client.R.d["meta"] = 1
            self.client.respond_to_server()
            self.client.shutdown()
            self.client = None
