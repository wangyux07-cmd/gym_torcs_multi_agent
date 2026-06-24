"""
gym_torcs_env.py
================
Gymnasium-compatible TORCS environment for PPO / IPPO training.

Architecture overview:
    TORCS (game window, you start manually)
        ↕  UDP on port 3001
    snakeoil3 Client (low-level socket wrapper, existing file)
        ↕  Python dicts
    TorcsEnv (this file -- translates dicts into Gym interface)
        ↕  numpy arrays
    PPO agent (stable-baselines3, separate training script)

Design decisions:
    - Windows-native: no os.system('pkill'), no shell scripts.
      You start TORCS manually before running the training script.
    - 65-dim state vector: same structure for single-car and multi-car.
      Single-car fills opponent sensors with max-range values (200m).
    - 3-dim continuous action: steer, accel, brake. Gear is rule-based.
    - Composite reward with configurable weights from config.py.
    - Per-episode CSV logging for thesis data collection.

Prerequisites:
    pip install gymnasium numpy
    (stable-baselines3 and torch are needed for training, not for this file)

Usage:
    1. Start TORCS → Race → Practice → select track → scr_server → New Race
    2. In Python:
        from gym_torcs_env import TorcsEnv
        env = TorcsEnv()
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
"""

import math
import os
import csv
import time
import numpy as np
from datetime import datetime

# --- Gymnasium import (required for SB3 v2+) ---
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "gymnasium is required. Install it with: pip install gymnasium"
    )

# --- Project imports ---
import config

try:
    from snakeoil3_gym import Client
except ImportError:
    try:
        from torcs_jm_par import Client
    except ImportError:
        raise ImportError(
            "Cannot find the TORCS UDP Client class. "
            "Make sure snakeoil3_gym.py (or torcs_jm_par.py) is in the same directory."
        )


# ================================================================
#  Constants
# ================================================================

NORM = {
    "angle":        math.pi,
    "trackPos":     2.0,
    "speedX":       300.0,
    "speedY":       50.0,
    "speedZ":       20.0,
    "track":        200.0,
    "wheelSpinVel": 100.0,
    "rpm":          10000.0,
    "opponents":    200.0,
}

GEAR_THRESHOLDS = [0, 20, 40, 80, 100, 180]


# ================================================================
#  Custom exception for crash detection
# ================================================================

class TorcsCrashError(Exception):
    """
    Raised when TORCS stops responding for too long, which almost
    always means the game has crashed (a known stability issue with
    this engine, especially after long sessions or many race restarts).
    """
    pass


# ================================================================
#  TorcsEnv
# ================================================================

class TorcsEnv(gym.Env):
    """
    Gymnasium environment wrapping TORCS via the SCR UDP protocol.

    Observation space: Box(-1, 1, shape=(65,))
        See _build_state_vector() for the exact layout.

    Action space: Box([-1, 0, 0], [1, 1, 1], shape=(3,))
        [0] steer  in [-1, 1]   (left / right)
        [1] accel  in [0, 1]    (throttle)
        [2] brake  in [0, 1]    (brake pedal)
    """

    metadata = {"render_modes": []}

    # ============================================================
    #  Initialization
    # ============================================================

    def __init__(self, port=None, host=None):
        super().__init__()

        self.port = port or config.TORCS_PARAMS["port"]
        self.host = host or config.TORCS_PARAMS["host"]

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.client = None
        self.time_step = 0
        self.prev_obs_raw = None
        self.prev_steer = None
        self.wall_pinned_counter = 0
        self.pinned_duration = 0
        self.prev_dist_from_start = None
        self.is_initial_reset = True

        self.episode_count = 0
        self._log_file = None
        self._log_writer = None

    # ============================================================
    #  reset()
    # ============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        while True:
            try:
                if not self.is_initial_reset and self.client is not None:
                    self._send_restart()

                self._disconnect()

                self.client = Client(p=self.port, H=self.host)
                self._safe_get_input(max_retries=10)
                raw_obs = self.client.S.d
                break

            except TorcsCrashError as e:
                self._wait_for_manual_restart(e)

        self.time_step = 0
        self.prev_obs_raw = dict(raw_obs)
        self.prev_dist_from_start = raw_obs.get("distFromStart", 0)
        self.prev_steer = None
        self.wall_pinned_counter = 0
        self.pinned_duration = 0
        self.is_initial_reset = False

        self._init_episode_log()

        state = self._build_state_vector(raw_obs)
        info = {"raw_obs": raw_obs}
        return state, info

    # ============================================================
    #  step()
    # ============================================================

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1],  0.0, 1.0))
        brake = float(np.clip(action[2],  0.0, 1.0))

        accel = self._apply_traction_control(accel)

        prior_obs = self.client.S.d
        ep = config.EPISODE_PARAMS
        if (abs(prior_obs.get("trackPos", 0.0)) > ep["wall_pinned_trackpos_threshold"]
                and prior_obs.get("speedX", 0.0) < ep["wall_pinned_speed_threshold"]):
            self.pinned_duration += 1
        else:
            self.pinned_duration = 0

        if self.pinned_duration >= ep["reverse_assist_time_limit"]:
            gear = -1
        else:
            gear = self._auto_gear(prior_obs.get("speedX", 0))

        self.client.R.d["steer"]  = steer
        self.client.R.d["accel"]  = accel
        self.client.R.d["brake"]  = brake
        self.client.R.d["gear"]   = gear
        self.client.R.d["clutch"] = 0
        self.client.R.d["meta"]   = 0

        try:
            self._safe_respond()
            self._safe_get_input(max_retries=10)
            raw_obs = self.client.S.d
        except TorcsCrashError as e:
            self._wait_for_manual_restart(e)
            self._close_episode_log()
            state = self._build_state_vector(self.prev_obs_raw or {})
            info = {
                "raw_obs": self.prev_obs_raw or {},
                "reward_info": {},
                "terminal_reason": "torcs_crash",
                "time_step": self.time_step,
            }
            return state, 0.0, False, True, info

        state = self._build_state_vector(raw_obs)
        reward, reward_info = self._compute_reward(raw_obs, steer)
        terminated, truncated, terminal_reason = self._check_terminal(raw_obs)

        terminal_reward = self._get_terminal_reward(terminal_reason)
        reward += terminal_reward
        reward_info["terminal"] = terminal_reward

        self._log_step(raw_obs, action, reward, reward_info, terminal_reason)

        if terminated or truncated:
            self._send_restart()
            self._close_episode_log()

        self.prev_obs_raw = dict(raw_obs)
        self.time_step += 1

        info = {
            "raw_obs": raw_obs,
            "reward_info": reward_info,
            "terminal_reason": terminal_reason,
            "time_step": self.time_step,
        }
        return state, reward, terminated, truncated, info

    # ============================================================
    #  State construction
    # ============================================================

    def _build_state_vector(self, raw):
        """
        Convert the raw TORCS sensor dictionary into a normalized
        65-dimensional numpy array.

        Layout (indices):
            [0]      angle           -- car heading vs track direction
            [1]      trackPos        -- lateral offset from track center
            [2]      speedX          -- longitudinal speed
            [3]      speedY          -- lateral speed
            [4]      speedZ          -- vertical speed
            [5:24]   track[0..18]    -- 19 range-finder distances
            [24:28]  wheelSpinVel[0..3] -- 4 wheel rotation speeds
            [28]     rpm             -- engine revs
            [29:65]  opponents[0..35]-- 36 opponent distance sensors
        """
        angle = raw.get("angle", 0.0) / NORM["angle"]
        track_pos = np.clip(raw.get("trackPos", 0.0), -2.0, 2.0) / NORM["trackPos"]
        speed_x = raw.get("speedX", 0.0) / NORM["speedX"]
        speed_y = raw.get("speedY", 0.0) / NORM["speedY"]
        speed_z = raw.get("speedZ", 0.0) / NORM["speedZ"]

        track = np.array(raw.get("track", [0.0] * 19), dtype=np.float32)
        track = track / NORM["track"]

        wsv = np.array(raw.get("wheelSpinVel", [0.0] * 4), dtype=np.float32)
        wsv = wsv / NORM["wheelSpinVel"]

        rpm = raw.get("rpm", 0.0) / NORM["rpm"]

        opp = np.array(raw.get("opponents", [200.0] * 36), dtype=np.float32)
        opp = opp / NORM["opponents"]

        state = np.concatenate([
            [angle, track_pos, speed_x, speed_y, speed_z],
            track,
            wsv,
            [rpm],
            opp,
        ]).astype(np.float32)

        return np.clip(state, -1.0, 1.0)

    # ============================================================
    #  Automatic gear shifting (rule-based, not learned by PPO)
    # ============================================================

    def _auto_gear(self, speed_x):
        gear = 1
        for i, threshold in enumerate(GEAR_THRESHOLDS):
            if speed_x > threshold:
                gear = i + 1
        return min(gear, 6)

    def _apply_traction_control(self, accel):
        """
        Dampens accel when the rear wheels are spinning noticeably faster
        than the front wheels (tire slip). Mirrors traction_control() from
        the original rule-based driver (torcs_jm_par.py).
        """
        tc = config.TRACTION_CONTROL
        if not tc["enabled"]:
            return accel

        wsv = self.client.S.d.get("wheelSpinVel", [0.0] * 4)
        if len(wsv) < 4:
            return accel

        rear_front_diff = (wsv[2] + wsv[3]) - (wsv[0] + wsv[1])
        if rear_front_diff > tc["wheel_spin_diff_threshold"]:
            accel -= tc["accel_reduction"]

        return max(0.0, accel)

    # ============================================================
    #  Reward: orchestrator
    # ============================================================

    def _compute_reward(self, obs, steer):
        """
        Compute the composite per-step reward.

        R_total = w_speed * R_speed
                + w_safety * R_safety
                + w_smooth * R_smooth
                + w_anticipation * R_anticipation
                + time_penalty
        """
        w = config.REWARD_WEIGHTS

        r_speed        = self._compute_speed_reward(obs)
        r_safety       = self._compute_safety_reward(obs)
        r_smooth       = self._compute_smoothness_reward(steer)
        r_anticipation = self._compute_anticipation_reward(obs)
        r_time         = config.REWARD_PARAMS["time_penalty"]

        total = (
            w["w_speed"]        * r_speed
            + w["w_safety"]     * r_safety
            + w["w_smooth"]     * r_smooth
            + w["w_anticipation"] * r_anticipation
            + r_time
        )

        info = {
            "r_speed":        r_speed,
            "r_safety":       r_safety,
            "r_smooth":       r_smooth,
            "r_anticipation": r_anticipation,
            "r_time":         r_time,
        }
        return total, info

    # ============================================================
    #  Reward: speed component
    # ============================================================

    def _compute_speed_reward(self, obs):
        """
        R_speed = speedX * cos(angle) - speedX * |sin(angle)| * k
        """
        speed_x = obs.get("speedX", 0.0)
        angle   = obs.get("angle", 0.0)
        k       = config.REWARD_PARAMS["lateral_penalty_k"]

        forward = speed_x * math.cos(angle)
        lateral = speed_x * abs(math.sin(angle)) * k

        return forward - lateral

    # ============================================================
    #  Reward: safety component
    # ============================================================

    def _compute_safety_reward(self, obs):
        """
        R_safety = -max(0, |trackPos| - safety_margin) ** 2
        Squared so the penalty grows fast near the track edge.
        """
        track_pos = abs(obs.get("trackPos", 0.0))
        margin    = config.REWARD_PARAMS["safety_margin"]

        if track_pos > margin:
            return -((track_pos - margin) ** 2)
        return 0.0

    # ============================================================
    #  Reward: steering smoothness
    # ============================================================

    def _compute_smoothness_reward(self, steer):
        """
        Penalizes large frame-to-frame changes in the steering action.
        Discourages abrupt full-lock corrections that cause spin-outs.
        """
        if self.prev_steer is None:
            delta = 0.0
        else:
            delta = abs(steer - self.prev_steer)
        self.prev_steer = steer
        k = config.REWARD_PARAMS["smoothness_k"]
        return -k * delta

    # ============================================================
    #  Reward: anticipation component
    # ============================================================

    def _compute_anticipation_reward(self, obs):
        """
        Penalizes maintaining high speed when the front-facing distance
        sensors show the track narrowing ahead (an approaching corner).

        Logic:
            1. Look at a forward-facing cone of sensors (indices 6-12,
               i.e. -30deg to +30deg) and take the MINIMUM reading --
               the most conservative estimate of how much open track is
               ahead before something (a wall, a corner) gets close.
            2. Convert that distance into a "safe speed" estimate:
               more clearance = higher safe speed, capped at a maximum
               so long straights are never penalized.
            3. If current speed exceeds that safe estimate, penalize the
               excess. If not, this term is zero -- it never tells the
               agent HOW to brake or steer, only that going this fast
               with this little clearance ahead is undesirable.

        Added after log analysis showed the car repeatedly entering
        corners at near-maximum speed with no signal discouraging this,
        since instantaneous safety reward only fires once already near
        the track edge -- often too late to avoid a collision.
        """
        p = config.REWARD_PARAMS
        track = obs.get("track", [200.0] * 19)
        indices = p["anticipation_sensor_indices"]
        sensors = [track[i] for i in indices if i < len(track)]
        forward_distance = min(sensors) if sensors else 200.0

        safe_speed = min(
            forward_distance * p["anticipation_speed_per_meter"],
            p["anticipation_max_safe_speed"],
        )

        speed_x = obs.get("speedX", 0.0)
        excess = max(0.0, speed_x - safe_speed)
        return -excess

    # ============================================================
    #  Reward: terminal (one-time, applied when episode ends)
    # ============================================================

    def _get_terminal_reward(self, reason):
        p = config.REWARD_PARAMS

        if reason == "collision":
            return p["collision_penalty"]
        elif reason == "out_of_track":
            return p["out_of_track_penalty"]
        elif reason == "stuck" or reason == "wall_pinned":
            return p["stuck_penalty"]
        elif reason == "backward":
            return p["backward_penalty"]
        elif reason == "lap_complete":
            return p["lap_complete_bonus"]
        return 0.0

    # ============================================================
    #  Termination logic
    # ============================================================

    def _check_terminal(self, obs):
        # --- Collision (damage-based) ---
        prev_damage = self.prev_obs_raw.get("damage", 0.0) if self.prev_obs_raw else 0.0
        if obs.get("damage", 0.0) - prev_damage > 0:
            return True, False, "collision"

        # --- Out of track ---
        if abs(obs.get("trackPos", 0.0)) > 1.0:
            return True, False, "out_of_track"

        # --- Running backward ---
        if math.cos(obs.get("angle", 0.0)) < 0:
            return True, False, "backward"

        # --- Wall-pinned ---
        ep = config.EPISODE_PARAMS
        track_pos_abs = abs(obs.get("trackPos", 0.0))
        if (track_pos_abs > ep["wall_pinned_trackpos_threshold"]
                and obs.get("speedX", 0.0) < ep["wall_pinned_speed_threshold"]):
            self.wall_pinned_counter += 1
            if self.wall_pinned_counter > ep["wall_pinned_time_limit"]:
                return True, False, "wall_pinned"
        else:
            self.wall_pinned_counter = 0

        # --- Stuck ---
        if self.time_step > ep["stuck_time_limit"]:
            if obs.get("speedX", 0.0) < ep["stuck_speed_threshold"]:
                return True, False, "stuck"

        # --- Lap completed ---
        dist = obs.get("distFromStart", 0.0)
        dist_raced = obs.get("distRaced", 0.0)
        if self.prev_dist_from_start is not None:
            delta = dist - self.prev_dist_from_start
            if delta < -1000:
                self.prev_dist_from_start = dist
                if dist_raced >= ep["min_dist_for_lap"]:
                    return True, False, "lap_complete"
                else:
                    pass
        self.prev_dist_from_start = dist

        # --- Step limit ---
        if self.time_step >= ep["max_steps"]:
            return False, True, "max_steps"

        return False, False, ""

    # ============================================================
    #  Logging
    # ============================================================

    def _init_episode_log(self):
        if not config.LOGGING["enabled"]:
            return

        log_dir = config.LOGGING["log_dir"]
        os.makedirs(log_dir, exist_ok=True)

        self.episode_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ep{self.episode_count:05d}_{timestamp}.csv"
        filepath = os.path.join(log_dir, filename)

        self._log_file = open(filepath, "w", newline="")
        self._log_writer = csv.writer(self._log_file)

        self._log_writer.writerow([
            "step",
            "speedX", "speedY", "angle", "trackPos",
            "distFromStart", "distRaced", "damage",
            "steer", "accel", "brake", "gear",
            "r_speed", "r_safety", "r_smooth", "r_anticipation", "r_time", "r_terminal", "reward_total",
            "terminal_reason",
        ])

    def _log_step(self, raw_obs, action, reward, reward_info, terminal_reason):
        if self._log_writer is None:
            return

        self._log_writer.writerow([
            self.time_step,
            raw_obs.get("speedX", 0),
            raw_obs.get("speedY", 0),
            raw_obs.get("angle", 0),
            raw_obs.get("trackPos", 0),
            raw_obs.get("distFromStart", 0),
            raw_obs.get("distRaced", 0),
            raw_obs.get("damage", 0),
            float(action[0]),
            float(action[1]),
            float(action[2]),
            self.client.R.d.get("gear", 0),
            reward_info.get("r_speed", 0),
            reward_info.get("r_safety", 0),
            reward_info.get("r_smooth", 0),
            reward_info.get("r_anticipation", 0),
            reward_info.get("r_time", 0),
            reward_info.get("terminal", 0),
            reward,
            terminal_reason,
        ])

    def _close_episode_log(self):
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
            self._log_writer = None

    # ============================================================
    #  Fault-tolerant communication with TORCS
    # ============================================================

    def _safe_get_input(self, max_retries=10):
        sock = self.client.so
        for attempt in range(max_retries):
            try:
                sockdata, _addr = sock.recvfrom(2 ** 17)
                sockdata = sockdata.decode("utf-8")
            except OSError:
                continue

            if not sockdata:
                continue
            if "***identified***" in sockdata:
                continue
            if "***shutdown***" in sockdata or "***restart***" in sockdata:
                raise TorcsCrashError(
                    "TORCS sent a shutdown/restart signal mid-step. "
                    "This usually means the race ended unexpectedly."
                )

            self.client.S.parse_server_str(sockdata)
            return

        raise TorcsCrashError(
            f"No response from TORCS after {max_retries} attempts "
            f"(~{max_retries} seconds). TORCS has likely crashed."
        )

    def _safe_respond(self):
        try:
            self.client.respond_to_server()
        except OSError as e:
            raise TorcsCrashError(f"Failed to send action to TORCS: {e}")

    def _wait_for_manual_restart(self, error):
        print("\n" + "=" * 60)
        print("  TORCS CONNECTION LOST")
        print("=" * 60)
        print(f"  Reason: {error}")
        print("  Please do the following:")
        print("    1. Close the crashed TORCS window if it is still open.")
        print("    2. Restart TORCS: Race -> Practice -> Configure Race")
        print("       -> select a track -> driver = scr_server -> New Race")
        print("    3. Wait for the blue 'waiting for connection' screen.")
        print("  Then press Enter here to resume.")
        print("=" * 60)
        input("  Press Enter once TORCS is ready... ")
        print("  Resuming...\n")

    def _send_restart(self):
        if self.client is None:
            return
        try:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()
        except Exception:
            pass
        time.sleep(0.5)

    def _disconnect(self):
        if self.client is None:
            return
        try:
            self.client.shutdown()
        except Exception:
            pass
        self.client = None

    # ============================================================
    #  Cleanup
    # ============================================================

    def close(self):
        if self._log_writer is not None:
            self._log_writer.writerow(
                [self.time_step] + [""] * 18 + ["training_stopped"]
            )
        self._close_episode_log()
        self._disconnect()
        super().close()