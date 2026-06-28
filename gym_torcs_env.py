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

Responsibilities of THIS file
------------------------------
  - Gym spaces definition (observation / action).
  - UDP lifecycle: connect, send action, receive obs, reconnect on crash.
  - State vector construction (_build_state_vector).
  - Rule-based actuator helpers (_auto_gear, _apply_traction_control,
    reverse-assist logic).
  - Orchestration of the per-step loop: reward, termination, logging.

What this file deliberately does NOT contain
---------------------------------------------
  - Reward arithmetic  →  reward.py
  - Termination logic  →  termination.py
  - CSV logging        →  episode_logger.py

Design decisions:
    - Windows-native: no os.system('pkill'), no shell scripts.
      You start TORCS manually before running the training script.
    - 29-dim state vector (opponents removed -- constant noise in single-car
      training).
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
import time
import numpy as np

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
from reward import (
    EpisodeRewardState,
    RunRewardState,
    fresh_episode_state,
    compute_reward,
)
from termination import (
    TerminationState,
    fresh_termination_state,
    check_terminal,
)
from episode_logger import EpisodeLogger

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

# Normalization divisors: map raw sensor ranges to roughly [-1, 1] or [0, 1].
NORM = {
    "angle":        math.pi,   # radians, range [-pi, pi]
    "trackPos":     2.0,       # clip to [-2, 2] first, then /2 → [-1, 1]
    "speedX":       300.0,     # km/h
    "speedY":       50.0,      # km/h (lateral)
    "speedZ":       20.0,      # km/h (vertical)
    "track":        200.0,     # meters (19 range-finder sensors)
    "wheelSpinVel": 100.0,     # rad/s (4 wheels)
    "rpm":          10000.0,   # engine revolutions per minute
}

# Rule-based gear shifting thresholds (km/h).
GEAR_THRESHOLDS = [0, 20, 40, 80, 100, 180]


# ================================================================
#  Custom exception for crash detection
# ================================================================

class TorcsCrashError(Exception):
    """
    Raised when TORCS stops responding for too long, which almost
    always means the game has crashed (a known stability issue with
    this engine, especially after long sessions or many race restarts).

    Training scripts should catch this, pause for manual TORCS restart,
    and resume rather than letting the whole training run die.
    """
    pass


# ================================================================
#  TorcsEnv
# ================================================================

class TorcsEnv(gym.Env):
    """
    Gymnasium environment wrapping TORCS via the SCR UDP protocol.

    Observation space: Box(-1, 1, shape=(29,))
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
        Create the environment.  Does NOT connect to TORCS yet --
        the connection is established on the first reset() call.
        """
        super().__init__()

        self.port = port or config.TORCS_PARAMS["port"]
        self.host = host or config.TORCS_PARAMS["host"]

        # --- Gym spaces ---
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(config.STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # --- UDP client ---
        self.client = None
        self.is_initial_reset = True

        # --- Previous raw obs (needed by termination damage check) ---
        self.prev_obs_raw = None

        # ---- Episode-scoped state --------------------------------
        # These are reset by _reset_episode_state() at every episode start.
        self.ep_reward_state: EpisodeRewardState = fresh_episode_state()
        self.term_state:      TerminationState   = fresh_termination_state()
        self.pinned_duration: int  = 0   # Reverse-assist gear logic counter
        self.time_step:       int  = 0

        # ---- Run-scoped state (intentionally NOT reset each episode) ----
        # furthest_distance_ever is the single global high-water mark.
        # See reward.py RunRewardState for the known --resume limitation.
        self.run_reward_state: RunRewardState = RunRewardState()

        # --- Logging ---
        self.episode_count = 0
        self._logger: EpisodeLogger | None = None

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
            5. Return the normalized 29-dim observation + info dict.
        """
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

        # --- Close previous episode log (if any) ---
        if self._logger is not None:
            self._logger.close()

        # --- Reset all episode-scoped state ---
        self._reset_episode_state(raw_obs)

        # --- Open new episode log ---
        self.episode_count += 1
        self._logger = EpisodeLogger(self.episode_count)

        state = self._build_state_vector(raw_obs)
        info  = {"raw_obs": raw_obs}
        return state, info

    def _reset_episode_state(self, raw_obs: dict) -> None:
        """
        Reset all per-episode counters and state objects.

        IMPORTANT: run_reward_state (furthest_distance_ever) is NOT touched
        here -- it is deliberately persistent across episodes.
        """
        # Episode-scoped reward state (prev_steer, checkpoints_reached)
        self.ep_reward_state = fresh_episode_state()

        # Episode-scoped termination counters
        self.term_state = fresh_termination_state(
            initial_dist_from_start=raw_obs.get("distFromStart", 0)
        )

        # Reverse-assist counter
        self.pinned_duration = 0

        # Step counter
        self.time_step = 0

        # Snapshot of obs for the next step's damage-delta check
        # IMPORTANT: must copy, not alias -- client.S.d is mutated in place.
        self.prev_obs_raw = dict(raw_obs)

        self.is_initial_reset = False

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
            observation:  np.array shape (29,)
            reward:       float
            terminated:   bool
            truncated:    bool
            info:         dict
        """
        # ------ 1. Parse and clip action ------
        steer = float(np.clip(action[0], -1.0,  1.0))
        accel = float(np.clip(action[1],  0.0,  1.0))
        brake = float(np.clip(action[2],  0.0,  1.0))

        # ------ 1b. Traction control (deterministic, not learned) ------
        accel = self._apply_traction_control(accel)

        # ------ 1c. Reverse-assist gear logic ------
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

        # ------ 2. Write action to the client ------
        self.client.R.d["steer"]  = steer
        self.client.R.d["accel"]  = accel
        self.client.R.d["brake"]  = brake
        self.client.R.d["gear"]   = gear
        self.client.R.d["clutch"] = 0
        self.client.R.d["meta"]   = 0

        # ------ 3. Exchange with TORCS ------
        try:
            self._safe_respond()
            self._safe_get_input(max_retries=10)
            raw_obs = self.client.S.d
        except TorcsCrashError as e:
            self._wait_for_manual_restart(e)
            if self._logger is not None:
                self._logger.close()
                self._logger = None
            state = self._build_state_vector(self.prev_obs_raw or {})
            info  = {
                "raw_obs":        self.prev_obs_raw or {},
                "reward_info":    {},
                "terminal_reason": "torcs_crash",
                "time_step":       self.time_step,
            }
            return state, 0.0, False, True, info

        # ------ 4. Build normalized state ------
        state = self._build_state_vector(raw_obs)

        # ------ 5. Termination check (must come before reward so that
        #            terminal_reason is available to reward functions) ------
        terminated, truncated, terminal_reason, self.term_state = check_terminal(
            obs=raw_obs,
            prev_obs=self.prev_obs_raw,
            time_step=self.time_step,
            state=self.term_state,
        )

        # ------ 6. Compute composite reward ------
        reward, reward_info, self.ep_reward_state, self.run_reward_state = compute_reward(
            obs=raw_obs,
            steer=steer,
            accel=accel,
            brake=brake,
            ep_state=self.ep_reward_state,
            run_state=self.run_reward_state,
            terminal_reason=terminal_reason,
        )

        # ------ 7. Log this step ------
        if self._logger is not None:
            self._logger.log_step(
                time_step=self.time_step,
                raw_obs=raw_obs,
                action=action,
                reward=reward,
                reward_info=reward_info,
                gear=gear,
                terminal_reason=terminal_reason,
            )

        # ------ 8. If episode is over, signal TORCS and close log ------
        if terminated or truncated:
            self._send_restart()
            if self._logger is not None:
                self._logger.close()
                self._logger = None

        # ------ 9. Advance internal state ------
        # IMPORTANT: copy, not alias (client.S.d is mutated in place).
        self.prev_obs_raw = dict(raw_obs)
        self.time_step   += 1

        info = {
            "raw_obs":         raw_obs,
            "reward_info":     reward_info,
            "terminal_reason": terminal_reason,
            "time_step":       self.time_step,
        }
        return state, reward, terminated, truncated, info

    # ============================================================
    #  State construction
    # ============================================================

    def _build_state_vector(self, raw) -> np.ndarray:
        """
        Build the 29-dimensional normalized observation vector.

        Layout (indices):
            [0]      angle           -- car heading vs track direction
            [1]      trackPos        -- lateral offset from track center
            [2]      speedX          -- longitudinal speed
            [3]      speedY          -- lateral speed
            [4]      speedZ          -- vertical speed
            [5:24]   track[0..18]    -- 19 range-finder distances
            [24:28]  wheelSpinVel[0..3]
            [28]     rpm
        """
        angle     = raw.get("angle",    0.0) / NORM["angle"]
        track_pos = np.clip(raw.get("trackPos", 0.0), -2.0, 2.0) / NORM["trackPos"]
        speed_x   = raw.get("speedX",   0.0) / NORM["speedX"]
        speed_y   = raw.get("speedY",   0.0) / NORM["speedY"]
        speed_z   = raw.get("speedZ",   0.0) / NORM["speedZ"]

        track = np.array(raw.get("track", [0.0] * 19), dtype=np.float32) / NORM["track"]
        wsv   = np.array(raw.get("wheelSpinVel", [0.0] * 4), dtype=np.float32) / NORM["wheelSpinVel"]
        rpm   = raw.get("rpm", 0.0) / NORM["rpm"]

        state = np.concatenate([
            [angle, track_pos, speed_x, speed_y, speed_z],  # 5
            track,                                           # 19
            wsv,                                             # 4
            [rpm],                                           # 1
        ]).astype(np.float32)                                # total = 29

        return np.clip(state, -1.0, 1.0)

    # ============================================================
    #  Rule-based actuator helpers (not learned by PPO)
    # ============================================================

    def _auto_gear(self, speed_x: float) -> int:
        """Simple threshold-based gear shifting."""
        gear = 1
        for i, threshold in enumerate(GEAR_THRESHOLDS):
            if speed_x > threshold:
                gear = i + 1
        return min(gear, 6)

    def _apply_traction_control(self, accel: float) -> float:
        """
        Dampen accel when rear wheels spin noticeably faster than front
        wheels (tire slip).  Mirrors the original rule-based driver's
        traction_control() from torcs_jm_par.py.
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
    #  Fault-tolerant communication with TORCS
    # ============================================================

    def _safe_get_input(self, max_retries=10) -> None:
        """
        Bounded-retry replacement for client.get_servers_input().
        Raises TorcsCrashError after max_retries consecutive timeouts.
        """
        sock = self.client.so
        for _ in range(max_retries):
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
                    "TORCS sent a shutdown/restart signal mid-step."
                )
            self.client.S.parse_server_str(sockdata)
            return

        raise TorcsCrashError(
            f"No response from TORCS after {max_retries} attempts. "
            "TORCS has likely crashed."
        )

    def _safe_respond(self) -> None:
        """Guarded version of client.respond_to_server()."""
        try:
            self.client.respond_to_server()
        except OSError as e:
            raise TorcsCrashError(f"Failed to send action to TORCS: {e}")

    def _wait_for_manual_restart(self, error) -> None:
        """Pause and prompt the user to manually restart TORCS."""
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

    def _send_restart(self) -> None:
        """Send meta=1 telling TORCS to restart the race."""
        if self.client is None:
            return
        try:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()
        except Exception:
            pass
        time.sleep(0.5)

    def _disconnect(self) -> None:
        """Close the UDP socket if one is open."""
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

    def close(self) -> None:
        """Release all resources."""
        if self._logger is not None:
            self._logger.log_training_stopped(self.time_step)
            self._logger.close()
            self._logger = None
        self._disconnect()
        super().close()
