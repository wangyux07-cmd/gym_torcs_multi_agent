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
        # Fallback: some project layouts put the Client in a different module
        from torcs_jm_par import Client
    except ImportError:
        raise ImportError(
            "Cannot find the TORCS UDP Client class. "
            "Make sure snakeoil3_gym.py (or torcs_jm_par.py) is in the same directory."
        )


# ================================================================
#  Constants
# ================================================================

# Normalization divisors: map raw sensor ranges to roughly [-1, 1] or [0, 1].
# These are the approximate maximum magnitudes for each sensor.
NORM = {
    "angle":        math.pi,   # radians, range [-pi, pi]
    "trackPos":     2.0,       # clip to [-2, 2] first, then /2 → [-1, 1]
    "speedX":       300.0,     # km/h
    "speedY":       50.0,      # km/h (lateral)
    "speedZ":       20.0,      # km/h (vertical)
    "track":        200.0,     # meters (19 range-finder sensors)
    "wheelSpinVel": 100.0,     # rad/s (4 wheels)
    "rpm":          10000.0,   # engine revolutions per minute
    "opponents":    200.0,     # meters (36 opponent sensors)
}

# Rule-based gear shifting thresholds (km/h).
# Index 0 → gear 1, index 1 → gear 2, etc.
GEAR_THRESHOLDS = [0, 20, 40, 80, 100, 180]


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

    metadata = {"render_modes": []}  # TORCS renders its own window

    # ============================================================
    #  Initialization
    # ============================================================

    def __init__(self, port=None, host=None):
        """
        Create the environment. Does NOT connect to TORCS yet --
        the connection is established on the first reset() call.

        Args:
            port: TORCS server port (default from config.py)
            host: TORCS server host (default from config.py)
        """
        super().__init__()

        # Connection parameters
        self.port = port or config.TORCS_PARAMS["port"]
        self.host = host or config.TORCS_PARAMS["host"]

        # --- Define Gym spaces ---
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

        # --- Internal bookkeeping ---
        self.client = None          # snakeoil3 Client instance
        self.time_step = 0          # Steps elapsed in current episode
        self.prev_obs_raw = None    # Previous frame's raw sensor dict
        self.prev_dist_from_start = None  # For lap-completion detection
        self.is_initial_reset = True      # True until first reset() finishes

        # --- Logging ---
        self.episode_count = 0
        self._log_file = None
        self._log_writer = None

    # ============================================================
    #  reset()
    # ============================================================

    def reset(self, seed=None, options=None):
        """
        Start a new episode.

        Workflow:
            1. If a previous episode exists, tell TORCS to restart the race.
            2. Close the old UDP socket.
            3. Open a new UDP connection (new Client).
            4. Read the initial sensor data.
            5. Return the normalized 65-dim observation + info dict.

        IMPORTANT: TORCS must already be running and showing the
        "waiting for connection" screen before you call this.
        """
        super().reset(seed=seed)

        # --- Tell TORCS to restart (skip on the very first call) ---
        if not self.is_initial_reset and self.client is not None:
            self._send_restart()

        # --- Close old connection ---
        self._disconnect()

        # --- Open new connection ---
        print("    [DEBUG] reset: opening new Client connection...")
        self.client = Client(p=self.port, H=self.host)
        print("    [DEBUG] reset: Client created, calling get_servers_input()...")
        self.client.get_servers_input()
        print("    [DEBUG] reset: got initial sensor data.")
        raw_obs = self.client.S.d

        # --- Reset episode state ---
        self.time_step = 0
        self.prev_obs_raw = raw_obs
        self.prev_dist_from_start = raw_obs.get("distFromStart", 0)
        self.is_initial_reset = False

        # --- Start episode log ---
        self._init_episode_log()

        # --- Build observation ---
        state = self._build_state_vector(raw_obs)
        info = {"raw_obs": raw_obs}
        return state, info

    # ============================================================
    #  step()
    # ============================================================

    def step(self, action):
        """
        Execute one control step.

        Args:
            action: numpy array of shape (3,)
                [0] steer  in [-1, 1]
                [1] accel  in [0, 1]
                [2] brake  in [0, 1]

        Returns:
            observation:  np.array shape (65,)
            reward:       float
            terminated:   bool (episode ended by game logic)
            truncated:    bool (episode ended by time limit)
            info:         dict with debugging / logging data
        """
        # ------ 1. Parse and clip action ------
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1],  0.0, 1.0))
        brake = float(np.clip(action[2],  0.0, 1.0))

        # ------ 2. Write action to the client ------
        self.client.R.d["steer"]  = steer
        self.client.R.d["accel"]  = accel
        self.client.R.d["brake"]  = brake
        self.client.R.d["gear"]   = self._auto_gear(
            self.client.S.d.get("speedX", 0)
        )
        self.client.R.d["clutch"] = 0
        self.client.R.d["meta"]   = 0

        # ------ 3. Exchange with TORCS ------
        self.client.respond_to_server()
        self.client.get_servers_input()
        raw_obs = self.client.S.d

        # ------ 4. Build normalized state ------
        state = self._build_state_vector(raw_obs)

        # ------ 5. Compute per-step reward ------
        reward, reward_info = self._compute_reward(raw_obs)

        # ------ 6. Check termination ------
        terminated, truncated, terminal_reason = self._check_terminal(raw_obs)

        # ------ 7. Add one-time terminal reward if episode is ending ------
        terminal_reward = self._get_terminal_reward(terminal_reason)
        reward += terminal_reward
        reward_info["terminal"] = terminal_reward

        # ------ 8. Log this step ------
        self._log_step(raw_obs, action, reward, reward_info, terminal_reason)

        # ------ 9. If episode is over, tell TORCS to restart ------
        if terminated or truncated:
            self._send_restart()
            self._close_episode_log()

        # ------ 10. Advance internal state ------
        self.prev_obs_raw = raw_obs
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

        # Single-car: opponents will all be 200 (no other car nearby).
        # After /200 they become 1.0, which the network learns to ignore.
        opp = np.array(raw.get("opponents", [200.0] * 36), dtype=np.float32)
        opp = opp / NORM["opponents"]

        state = np.concatenate([
            [angle, track_pos, speed_x, speed_y, speed_z],  # 5
            track,                                           # 19
            wsv,                                             # 4
            [rpm],                                           # 1
            opp,                                             # 36
        ]).astype(np.float32)                                # total = 65

        return np.clip(state, -1.0, 1.0)

    # ============================================================
    #  Automatic gear shifting (rule-based, not learned by PPO)
    # ============================================================

    def _auto_gear(self, speed_x):
        """
        Simple rule: shift up when speed exceeds the threshold for that gear.
        Uses the same thresholds as the user's existing drive_modular().
        """
        gear = 1
        for i, threshold in enumerate(GEAR_THRESHOLDS):
            if speed_x > threshold:
                gear = i + 1
        return min(gear, 6)

    # ============================================================
    #  Reward: orchestrator
    # ============================================================

    def _compute_reward(self, obs):
        """
        Compute the composite per-step reward.

        R_total = w_speed * R_speed + w_safety * R_safety + w_ttc * R_ttc

        Returns:
            total_reward: float
            info_dict:    dict of individual components (for logging)
        """
        w = config.REWARD_WEIGHTS

        r_speed  = self._compute_speed_reward(obs)
        r_safety = self._compute_safety_reward(obs)
        r_ttc    = self._compute_ttc_reward(obs)

        total = (
            w["w_speed"]  * r_speed
            + w["w_safety"] * r_safety
            + w["w_ttc"]    * r_ttc
        )

        info = {
            "r_speed":  r_speed,
            "r_safety": r_safety,
            "r_ttc":    r_ttc,
        }
        return total, info

    # ============================================================
    #  Reward: speed component
    # ============================================================

    def _compute_speed_reward(self, obs):
        """
        R_speed = speedX * cos(angle) - speedX * |sin(angle)| * k

        First term:  reward forward progress along the track direction.
                     If the car is heading straight, cos(angle) ≈ 1, full credit.
                     If sideways, cos → 0, reduced credit.
        Second term: penalize lateral "snaking" (weaving left-right).
                     Prevents the agent from gaming the reward by oscillating.
        """
        speed_x = obs.get("speedX", 0.0)
        angle   = obs.get("angle", 0.0)
        k       = config.REWARD_PARAMS["lateral_penalty_k"]

        forward  = speed_x * math.cos(angle)
        lateral  = speed_x * abs(math.sin(angle)) * k

        return forward - lateral

    # ============================================================
    #  Reward: safety component
    # ============================================================

    def _compute_safety_reward(self, obs):
        """
        Single-car stage:
            R_safety = -max(0, |trackPos| - safety_margin)
            Starts penalizing when the car wanders past 80% of track width.
            This teaches the car to leave a safety buffer, not hug the edge.

        Multi-car stage (future):
            Will be extended to incorporate opponent proximity penalty.
        """
        track_pos = abs(obs.get("trackPos", 0.0))
        margin    = config.REWARD_PARAMS["safety_margin"]

        if track_pos > margin:
            return -(track_pos - margin)
        return 0.0

    # ============================================================
    #  Reward: Time-to-Collision component (placeholder)
    # ============================================================

    def _compute_ttc_reward(self, obs):
        """
        Time-to-Collision (TTC) penalty.

        Single-car stage:
            Returns 0. With w_ttc=0 in config, this is also multiplied by 0,
            so it has zero effect on training. The function exists so the
            code structure is ready for multi-car without refactoring.

        Multi-car stage (future implementation):
            1. Find the nearest opponent from the 36 opponent sensors.
            2. Estimate closing speed = own speed - relative approach rate.
            3. TTC = nearest_distance / closing_speed (if closing).
            4. Return a negative penalty that increases as TTC decreases,
               following Krasowski et al. (2022) safe-RL framework.
        """
        return 0.0

    # ============================================================
    #  Reward: terminal (one-time, applied when episode ends)
    # ============================================================

    def _get_terminal_reward(self, reason):
        """
        One-shot reward/penalty at the moment the episode terminates.
        Only called when terminated=True or truncated=True.
        """
        p = config.REWARD_PARAMS

        if reason == "out_of_track":
            return p["out_of_track_penalty"]
        elif reason == "stuck":
            return p["stuck_penalty"]
        elif reason == "backward":
            return p["backward_penalty"]
        elif reason == "lap_complete":
            return p["lap_complete_bonus"]
        # "max_steps" (truncated) or empty reason: no extra reward
        return 0.0

    # ============================================================
    #  Termination logic
    # ============================================================

    def _check_terminal(self, obs):
        """
        Determine whether the current episode should end.

        Returns:
            terminated: bool -- ended by game logic (crash, stuck, lap done)
            truncated:  bool -- ended by step limit
            reason:     str  -- human-readable label for logging / analysis
        """
        # --- Out of track ---
        if abs(obs.get("trackPos", 0.0)) > 1.0:
            return True, False, "out_of_track"

        # --- Running backward ---
        if math.cos(obs.get("angle", 0.0)) < 0:
            return True, False, "backward"

        # --- Stuck (too slow for too long) ---
        ep = config.EPISODE_PARAMS
        if self.time_step > ep["stuck_time_limit"]:
            if obs.get("speedX", 0.0) < ep["stuck_speed_threshold"]:
                return True, False, "stuck"

        # --- Lap completed ---
        # Detection: distFromStart jumps from a large value back near 0
        # when the car crosses the start/finish line.
        dist = obs.get("distFromStart", 0.0)
        if self.prev_dist_from_start is not None:
            delta = dist - self.prev_dist_from_start
            # A large negative jump means the car crossed the finish line.
            # Threshold -1000 avoids false triggers from small fluctuations.
            if delta < -1000:
                self.prev_dist_from_start = dist
                return True, False, "lap_complete"
        self.prev_dist_from_start = dist

        # --- Step limit ---
        if self.time_step >= ep["max_steps"]:
            return False, True, "max_steps"

        # --- Episode continues ---
        return False, False, ""

    # ============================================================
    #  Logging: per-episode CSV files for thesis data analysis
    # ============================================================

    def _init_episode_log(self):
        """Create a new CSV file for the current episode."""
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

        # CSV header -- each column maps to a thesis analysis metric
        self._log_writer.writerow([
            "step",
            # Raw sensor readings (for trajectory / behavior analysis)
            "speedX", "speedY", "angle", "trackPos",
            "distFromStart", "distRaced",
            # Actions taken by the agent
            "steer", "accel", "brake", "gear",
            # Reward breakdown (for ablation comparison)
            "r_speed", "r_safety", "r_ttc", "r_terminal", "reward_total",
            # Episode event
            "terminal_reason",
        ])

    def _log_step(self, raw_obs, action, reward, reward_info, terminal_reason):
        """Write one row to the episode CSV."""
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
            float(action[0]),
            float(action[1]),
            float(action[2]),
            self.client.R.d.get("gear", 0),
            reward_info.get("r_speed", 0),
            reward_info.get("r_safety", 0),
            reward_info.get("r_ttc", 0),
            reward_info.get("terminal", 0),
            reward,
            terminal_reason,
        ])

    def _close_episode_log(self):
        """Flush and close the current episode's CSV file."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
            self._log_writer = None

    # ============================================================
    #  Connection helpers
    # ============================================================

    def _send_restart(self):
        """Send the meta=1 signal telling TORCS to restart the race."""
        if self.client is None:
            return
        print("    [DEBUG] _send_restart: sending meta=1 to TORCS...")
        try:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()
            print("    [DEBUG] _send_restart: meta=1 sent successfully.")
        except Exception as e:
            print(f"    [DEBUG] _send_restart: exception during send: {e}")
        time.sleep(0.5)  # Give TORCS a moment to process the restart.
        print("    [DEBUG] _send_restart: done sleeping, returning.")

    def _disconnect(self):
        """Close the UDP socket if one is open."""
        if self.client is None:
            return
        print("    [DEBUG] _disconnect: closing socket...")
        try:
            self.client.shutdown()
            print("    [DEBUG] _disconnect: socket closed successfully.")
        except Exception as e:
            print(f"    [DEBUG] _disconnect: exception during shutdown: {e}")
        self.client = None

    # ============================================================
    #  Cleanup (called when training ends)
    # ============================================================

    def close(self):
        """Release all resources."""
        self._close_episode_log()
        self._disconnect()
        super().close()