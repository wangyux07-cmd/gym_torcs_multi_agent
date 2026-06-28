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
        self.prev_steer = None      # Previous step's steer action (for smoothness penalty)
        self.prev_progress_potential = None  # Previous step's progress potential (for R_progress shaping)
        self.wall_pinned_counter = 0  # Consecutive steps spent jammed against the wall (for termination)
        self.out_of_track_counter = 0  # Consecutive steps off-track (grace period before termination)
        self.in_track_streak = 0       # Consecutive steps back IN bounds (must sustain this to clear the counter)
        # PERSISTENT across episodes (deliberately NOT reset in reset()
        # -- this is a single running high-water mark for the whole
        # training run, see _compute_record_reward). KNOWN LIMITATION:
        # this is in-memory only, not saved/restored with model
        # checkpoints -- if you stop and --resume training later, this
        # restarts from 0, meaning the first new max after a resume will
        # likely trigger a (probably undeserved, since it's not a TRUE
        # all-time record) payout. Acceptable given time constraints,
        # but worth knowing if you see an unexplained reward spike right
        # after a --resume.
        self.furthest_distance_ever = 0.0
        self.checkpoints_reached = set()  # Which distance checkpoints (config.REWARD_PARAMS["checkpoints"]) have already paid out this episode
        self.low_speed_counter = 0  # Consecutive steps below stuck_speed_threshold (resets the moment speed recovers)
        self.pinned_duration = 0      # Separate counter used to trigger reverse-gear assist
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

        If TORCS has crashed, this will detect it, pause and prompt you
        to manually restart TORCS, then retry automatically once you
        confirm it is ready.

        IMPORTANT: TORCS must already be running and showing the
        "waiting for connection" screen before you call this.
        """
        super().reset(seed=seed)

        while True:  # Retry loop: keeps trying until a connection succeeds
            try:
                # --- Tell TORCS to restart (skip on the very first call) ---
                if not self.is_initial_reset and self.client is not None:
                    self._send_restart()

                # --- Close old connection ---
                self._disconnect()

                # --- Open new connection ---
                self.client = Client(p=self.port, H=self.host)
                self._safe_get_input(max_retries=10)
                raw_obs = self.client.S.d
                break  # Success -- exit the retry loop

            except TorcsCrashError as e:
                self._wait_for_manual_restart(e)
                # Loop back and try again from scratch.

        # --- Reset episode state ---
        self.time_step = 0
        # IMPORTANT: must copy, not alias. self.client.S.d is mutated
        # in place by snakeoil3's parse_server_str() on every frame, so
        # storing a bare reference here would make prev_obs_raw silently
        # track the SAME dict as the current obs on the next step,
        # permanently breaking any "did this value change?" comparison
        # (this is exactly what made damage-based collision detection
        # never fire, despite damage clearly rising in the logs).
        self.prev_obs_raw = dict(raw_obs)
        self.prev_dist_from_start = raw_obs.get("distFromStart", 0)
        self.prev_steer = None
        self.prev_progress_potential = None
        self.wall_pinned_counter = 0
        self.out_of_track_counter = 0
        self.in_track_streak = 0
        self.checkpoints_reached = set()
        self.low_speed_counter = 0
        self.pinned_duration = 0
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

        # ------ 1b. Traction control filter (deterministic, not learned) ------
        # Same actuator-level safety net as gear shifting: dampens accel
        # when the rear wheels are spinning noticeably faster than the
        # front ones (tire slip), rather than leaving full wheelspin
        # unchecked. This is a physical safety filter, not a strategic
        # choice the agent needs to learn, so it's applied here just like
        # the rule-based driver's original traction_control() did.
        accel = self._apply_traction_control(accel)

        # ------ 1c. Reverse-assist gear logic (no override of steer/accel) ------
        # Uses the sensor reading from BEFORE this step's action (i.e. the
        # state the agent is currently reacting to) to decide whether the
        # car has been wall-pinned long enough to switch to reverse gear.
        # The agent's own steer/accel values are untouched -- only the
        # gear they get applied through changes, so if the agent steers
        # away from the wall while this fires, it will genuinely
        # experience successful reversing as a result of its OWN choice.
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

        # ------ 3. Exchange with TORCS (crash-safe) ------
        try:
            self._safe_respond()
            self._safe_get_input(max_retries=10)
            raw_obs = self.client.S.d
        except TorcsCrashError as e:
            self._wait_for_manual_restart(e)
            # We cannot continue this episode -- TORCS lost its state.
            # Tell the caller this episode is truncated (not a normal
            # termination) so the training script knows to call reset().
            self._close_episode_log()
            # Return the last known good observation as a placeholder;
            # the training script must call reset() immediately after this.
            state = self._build_state_vector(self.prev_obs_raw or {})
            info = {
                "raw_obs": self.prev_obs_raw or {},
                "reward_info": {},
                "terminal_reason": "torcs_crash",
                "time_step": self.time_step,
            }
            return state, 0.0, False, True, info

        # ------ 4. Build normalized state ------
        state = self._build_state_vector(raw_obs)

        # ------ 5. Compute per-step reward ------
        reward, reward_info = self._compute_reward(raw_obs, steer, accel, brake)

        # ------ 6. Check termination ------
        terminated, truncated, terminal_reason = self._check_terminal(raw_obs)

        # ------ 7. Add one-time terminal reward if episode is ending ------
        terminal_reward = self._get_terminal_reward(terminal_reason)
        reward += terminal_reward
        reward_info["terminal"] = terminal_reward

        # ------ 7b. Add one-time checkpoint reward (sub-goal, see config) ------
        checkpoint_reward = self._compute_checkpoint_reward(raw_obs, terminal_reason)
        reward += checkpoint_reward
        reward_info["checkpoint"] = checkpoint_reward

        # ------ 7c. Add one-time record-breaking reward (new global max distance) ------
        record_reward = self._compute_record_reward(raw_obs, terminal_reason)
        reward += record_reward
        reward_info["record"] = record_reward

        # ------ 8. Log this step ------
        self._log_step(raw_obs, action, reward, reward_info, terminal_reason)

        # ------ 9. If episode is over, tell TORCS to restart ------
        if terminated or truncated:
            self._send_restart()
            self._close_episode_log()

        # ------ 10. Advance internal state ------
        # Must copy (see note in reset()): client.S.d is mutated in place,
        # so storing a bare reference here makes prev_obs_raw silently
        # become the SAME object as next step's raw_obs.
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

    def _apply_traction_control(self, accel):
        """
        Dampens accel when the rear wheels are spinning noticeably faster
        than the front wheels (a sign of tire slip / wheelspin rather than
        actual grip). Mirrors the traction_control() logic from the
        original rule-based driver (torcs_jm_par.py).
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

    def _compute_reward(self, obs, steer, accel, brake):
        """
        Compute the composite per-step reward.

        R_total = w_speed * R_speed + w_safety * R_safety
                  + w_smooth * R_smooth + w_anticipation * R_anticipation
                  + w_progress * R_progress + w_pedal_conflict * R_pedal_conflict
                  + time_penalty

        The time_penalty is a flat, unweighted constant applied every step
        (see config.py REWARD_PARAMS["time_penalty"] for why).

        Returns:
            total_reward: float
            info_dict:    dict of individual components (for logging)
        """
        w = config.REWARD_WEIGHTS

        r_speed          = self._compute_speed_reward(obs)
        r_safety         = self._compute_safety_reward(obs)
        r_smooth         = self._compute_smoothness_reward(steer)
        r_anticipation   = self._compute_anticipation_reward(obs)
        r_progress       = self._compute_progress_reward(obs)
        r_pedal_conflict = self._compute_pedal_conflict_reward(accel, brake)
        r_time           = config.REWARD_PARAMS["time_penalty"]

        total = (
            w["w_speed"]  * r_speed
            + w["w_safety"] * r_safety
            + w["w_smooth"] * r_smooth
            + w["w_anticipation"] * r_anticipation
            + w["w_progress"] * r_progress
            + w["w_pedal_conflict"] * r_pedal_conflict
            + r_time
        )

        info = {
            "r_speed":  r_speed,
            "r_pedal_conflict": r_pedal_conflict,
            "r_safety": r_safety,
            "r_smooth": r_smooth,
            "r_anticipation": r_anticipation,
            "r_progress": r_progress,
            "r_time":   r_time,
        }
        return total, info

    # ============================================================
    #  Reward: potential-based "progress" component (NEW)
    # ============================================================

    def _compute_progress_reward(self, obs):
        """
        R_progress = gamma * Phi(s') - Phi(s)
        where Phi(s) = distRaced - lateral_k*|trackPos| - angle_k*|angle|

        ADDED to directly address a confirmed observed problem (visually
        verified by the user, not just inferred): the car DOES decelerate
        approaching the known hairpin hotspots (400-500m, 1200-1400m),
        but still fails -- meaning the car is willing to attempt
        corrections, but the existing reward structure actively punishes
        deceleration via R_speed dropping, which may be discouraging it
        from braking earlier/harder than it currently does.

        This term credits genuine moment-to-moment improvement in
        overall driving state (more distance covered, closer to center,
        better-aligned heading) rather than raw speed alone -- so
        slowing down to correct trackPos/angle can still register as
        "progress" even while R_speed drops, instead of being a pure
        loss with no offsetting signal.

        HONEST LIMITATION: if the true bottleneck at these hotspots is
        insufficient reaction distance/sensor warning (a geometry-level
        constraint), this term cannot manufacture more reaction space --
        it can only make the agent MORE WILLING to use the space it
        already has more decisively. Re-verify with
        check_approach_behavior.py after training with this enabled to
        see whether deceleration becomes earlier/stronger, and whether
        that's enough.

        Uses the same potential-based shaping form (Ng, Harada &
        Russell, 1999) as the existing recovery-reward design elsewhere
        in this project: telescopes to ~zero over any round trip, so
        there's no way to farm this by oscillating state back and
        forth -- only genuine net improvement over a stretch pays out.
        """
        p = config.REWARD_PARAMS
        dist_raced = obs.get("distRaced", 0.0)
        track_pos = obs.get("trackPos", 0.0)
        angle = obs.get("angle", 0.0)

        potential_now = (
            dist_raced
            - p["progress_lateral_k"] * abs(track_pos)
            - p["progress_angle_k"] * abs(angle)
        )

        if self.prev_progress_potential is None:
            self.prev_progress_potential = potential_now
            return 0.0

        gamma = config.PPO_PARAMS["gamma"]
        shaping = gamma * potential_now - self.prev_progress_potential
        self.prev_progress_potential = potential_now
        return shaping


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
            R_safety = -max(0, |trackPos| - safety_margin)^2
                       - beta * speedX * |trackPos|

            The first term (squared margin penalty, unchanged) fires
            only once the car is meaningfully off-center, and grows
            fast as it gets closer to the edge.

            The second term (speed x deviation, new) is always active,
            scaling the deviation penalty by current speed -- drifting
            off-center at high speed is penalized more than drifting at
            low speed, even before crossing the safety_margin threshold.
            Added based on a TORCS-RL literature pattern (see config.py
            REWARD_PARAMS comment for the citation and caveats); this
            complements the angle-based lateral term already in
            _compute_speed_reward (heading mismatch) and the
            forward-clearance-based anticipation term (distance ahead)
            -- three different signals, not redundant with each other.

        Multi-car stage (future):
            Will be extended to incorporate opponent proximity penalty.
        """
        track_pos = abs(obs.get("trackPos", 0.0))
        speed_x   = obs.get("speedX", 0.0)
        margin    = config.REWARD_PARAMS["safety_margin"]
        beta      = config.REWARD_PARAMS["safety_speed_beta"]

        if track_pos > margin:
            boundary_penalty = -((track_pos - margin) ** 2)
        else:
            boundary_penalty = 0.0

        speed_deviation_penalty = -beta * speed_x * track_pos

        return boundary_penalty + speed_deviation_penalty

    # ============================================================
    def _compute_smoothness_reward(self, steer):
        """
        Penalizes large frame-to-frame changes in the steering action.

        Without this, nothing discourages the network from slamming the
        wheel from one extreme to the other -- which is exactly the
        "learns to correct away from the wall, but overcorrects and spins
        out the other side" behavior observed in training. This nudges
        the policy toward gradual, controlled steering adjustments instead
        of abrupt full-lock corrections.
        """
        if self.prev_steer is None:
            delta = 0.0
        else:
            delta = abs(steer - self.prev_steer)
        self.prev_steer = steer
        k = config.REWARD_PARAMS["smoothness_k"]
        return -k * delta

    def _compute_pedal_conflict_reward(self, accel, brake):
        """
        Penalizes pressing accel and brake at the same time, scaled by
        how much they overlap.

        ADDED after direct log evidence: in early-training episodes
        where the car barely moved (e.g. 2.96m over 800 steps), accel
        was frequently high (often >0.9) at the SAME step brake was also
        high (sometimes also >0.9) -- the two were fighting each other,
        wasting the car's forward effort rather than the car "not
        wanting" to move. This is a distinct problem from a weak speed
        incentive (raising w_speed doesn't fix two opposing forces
        canceling out; it would just make the car push harder into both
        pedals at once).

        Uses min(accel, brake) once both exceed the threshold, rather
        than a flat penalty for crossing it -- this scales the penalty
        to the actual size of the wasted overlap (accel=1.0,brake=1.0 is
        penalized far more than accel=0.9,brake=0.31, which barely
        crosses the threshold).
        """
        p = config.REWARD_PARAMS
        threshold = p["pedal_conflict_threshold"]
        if accel > threshold and brake > threshold:
            overlap = min(accel, brake)
            return -p["pedal_conflict_k"] * overlap
        return 0.0

    def _compute_anticipation_reward(self, obs):
        """
        Penalizes maintaining high speed when the front-facing distance
        sensors show the track narrowing ahead (an approaching corner).

        Logic:
            1. Look across the sensor sweep (now the FULL -90deg to
               +90deg range, all 19 sensors -- widened again from an
               earlier -50/+50 cone specifically to catch hairpin
               turns, where the apex wall first appears at wide angles
               while center sensors still read open track down the
               corner's "throat") and take the MINIMUM reading -- the
               most conservative estimate of how much open track is
               ahead.
            2. Convert that distance into a "safe speed" estimate using
               a sqrt relationship (safe_speed = k * sqrt(distance)),
               grounded in centripetal force physics: max cornering
               speed scales with sqrt(radius), not linearly with it.
               This is more conservative at larger clearances than the
               earlier linear formula, and capped at a maximum so long
               straights are never penalized.
            3. If current speed exceeds that safe estimate, penalize the
               excess. If not, this term is zero -- it never tells the
               agent HOW to brake or steer, only that going this fast
               with this little clearance ahead is undesirable.

        Revised after grid search data showed out_of_track as the
        dominant failure mode (~70% of episodes, vs ~5% collision), and
        raising w_safety made it WORSE, not better -- pointing at a
        detection-timing problem (the old narrow cone + linear formula
        let the car see corners too late and underestimated how much
        it needed to slow down), not a penalty-magnitude problem.
        """
        p = config.REWARD_PARAMS
        track = obs.get("track", [200.0] * 19)
        indices = p["anticipation_sensor_indices"]
        sensors = [track[i] for i in indices if i < len(track)]
        forward_distance = max(min(sensors) if sensors else 200.0, 0.0)

        safe_speed = min(
            p["anticipation_speed_per_sqrt_meter"] * math.sqrt(forward_distance),
            p["anticipation_max_safe_speed"],
        )

        speed_x = obs.get("speedX", 0.0)
        excess = max(0.0, speed_x - safe_speed)
        return -excess

    # ============================================================
    #  Reward: terminal (one-time, applied when episode ends)
    # ============================================================

    def _get_terminal_reward(self, reason):
        """
        One-shot reward/penalty at the moment the episode terminates.
        Only called when terminated=True or truncated=True.
        """
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
        # "max_steps" (truncated) or empty reason: no extra reward
        return 0.0

    def _compute_checkpoint_reward(self, obs, terminal_reason):
        """
        One-time bonus the first time distRaced crosses each configured
        checkpoint distance within an episode (config.REWARD_PARAMS
        ["checkpoints"]). Added because the per-step reward terms are
        all "direction"-shaped (reward fast/centered driving moment to
        moment) but nothing directly credits actually covering ground
        on this specific track -- giving the agent achievable
        intermediate sub-goals on the way to a full lap, scaled to fit
        this project's existing terminal-reward convention (tens, with
        lap_complete_bonus=150 as the ceiling -- NOT the much larger
        "thousands" scale used in a separate, unrelated experiment
        branch of this project).

        Checkpoints are placed at 25/50/75% of the known Corkscrew lap
        length (3608.45m -> ~900/1800/2700m), so the first one already
        requires getting through the early part of the track -- including
        whatever corner is currently causing the car to stop short.

        Guards against the same "farm it by oscillating across the
        line" exploit class as the out_of_track grace-period logic: a
        checkpoint only pays out ONCE per episode (self.checkpoints_reached),
        and not at all if THIS step is simultaneously a collision/
        out_of_track termination (i.e. the car didn't actually clear it
        safely, it just happened to cross the distance threshold at the
        moment it crashed/left the track).
        """
        dist_raced = obs.get("distRaced", 0.0)
        total_bonus = 0.0

        if terminal_reason in ("collision", "out_of_track"):
            return 0.0

        for checkpoint_distance, bonus in config.REWARD_PARAMS["checkpoints"]:
            if checkpoint_distance in self.checkpoints_reached:
                continue
            if dist_raced >= checkpoint_distance:
                self.checkpoints_reached.add(checkpoint_distance)
                total_bonus += bonus

        return total_bonus

    def _compute_record_reward(self, obs, terminal_reason):
        """
        One-time bonus when distRaced reaches a NEW all-time high for
        this training run (not per-episode -- a single global
        high-water mark, see self.furthest_distance_ever, deliberately
        persistent across episodes).

        Directly serves the priority shift to "just finish a lap,
        speed doesn't matter": this rewards genuinely exploring further
        than the policy has EVER gotten before, regardless of how slow
        or fast it got there.

        Safe by construction against the oscillation-farming exploit
        found elsewhere in this project: the high-water mark only ever
        increases (line below), so the same ground can never pay out
        twice -- there's no way to "flicker" past a point repeatedly to
        re-trigger this.

        Requires beating the old record by at least
        record_min_improvement meters (not just any improvement,
        however tiny) -- without this, ordinary random fluctuation of a
        few centimeters past the previous max would constantly trigger
        trivial, noisy payouts that don't represent real progress.

        Does not pay out if this same step is a collision/out_of_track
        termination (consistent with the checkpoint reward's logic --
        crossing the line at the exact moment of crashing doesn't count
        as a genuine, safe breakthrough).
        """
        if terminal_reason in ("collision", "out_of_track"):
            return 0.0

        p = config.REWARD_PARAMS
        dist_raced = obs.get("distRaced", 0.0)

        if dist_raced < self.furthest_distance_ever + p["record_min_improvement"]:
            return 0.0

        self.furthest_distance_ever = dist_raced
        return p["record_reward_k"] * (dist_raced ** 0.5)

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
        # --- Collision (damage-based, checked first) ---
        # Direct physics-engine signal: TORCS increments 'damage' whenever
        # the car hits something, regardless of where the impact happens
        # geometrically. This catches impacts that trackPos misses -- log
        # analysis found a high-speed hit (92.7 -> 58.5 km/h in one step)
        # at a track location where trackPos never crossed the
        # out-of-track threshold, leaving the car to drag itself for ~100
        # more undetected steps before the generic stuck check finally
        # ended the episode.
        prev_damage = self.prev_obs_raw.get("damage", 0.0) if self.prev_obs_raw else 0.0
        if obs.get("damage", 0.0) - prev_damage > 0:
            return True, False, "collision"

        # --- Out of track (soft, grace-period -- no longer instant) ---
        # Previously terminated the instant |trackPos| > 1.0. Softened
        # because instant termination only teaches the agent to AVOID
        # ever crossing the line, never to RECOVER once it has -- and
        # combined with the other per-step penalties largely vanishing
        # at low speed (see anticipation/smoothness below), this was
        # part of what made "just stop before the corner" look like a
        # safer bet than attempting it. The car now gets
        # out_of_track_grace_steps consecutive steps to steer back onto
        # the track before the episode actually ends. The continuous
        # squared R_safety penalty still grows the further past 1.0 the
        # car drifts, so there's no incentive to linger off-track on
        # purpose -- this only removes the IMMEDIATE reset.
        #
        # Anti-flicker fix included from the start (not bolted on after
        # the fact): the off-track counter is NOT cleared by a single
        # in-bounds frame. The car must stay continuously within bounds
        # for out_of_track_recovery_streak_required steps before the
        # counter resets -- otherwise an agent could learn to "flicker"
        # back and forth across the boundary, never accumulating enough
        # consecutive off-track steps to trigger termination while still
        # effectively riding/cutting along the edge indefinitely.
        ep = config.EPISODE_PARAMS
        if abs(obs.get("trackPos", 0.0)) > 1.0:
            self.out_of_track_counter += 1
            self.in_track_streak = 0
            if self.out_of_track_counter > ep["out_of_track_grace_steps"]:
                return True, False, "out_of_track"
        else:
            self.in_track_streak += 1
            if self.in_track_streak >= ep["out_of_track_recovery_streak_required"]:
                self.out_of_track_counter = 0

        # --- Running backward ---
        if math.cos(obs.get("angle", 0.0)) < 0:
            return True, False, "backward"

        # --- Wall-pinned (fast timeout) ---
        # A much shorter timer than the generic stuck check below,
        # specifically for cars jammed against the track edge with
        # almost no speed -- a dead-end situation with no recovery
        # value, so we end it quickly instead of burning the full
        # stuck_time_limit. See config.py EPISODE_PARAMS for rationale.
        ep = config.EPISODE_PARAMS
        track_pos_abs = abs(obs.get("trackPos", 0.0))
        if (track_pos_abs > ep["wall_pinned_trackpos_threshold"]
                and obs.get("speedX", 0.0) < ep["wall_pinned_speed_threshold"]):
            self.wall_pinned_counter += 1
            if self.wall_pinned_counter > ep["wall_pinned_time_limit"]:
                return True, False, "wall_pinned"
        else:
            self.wall_pinned_counter = 0

        # --- Stuck (too slow for too long, CONSECUTIVELY) ---
        # FIXED: the previous version checked "has total time_step
        # passed stuck_time_limit AND is speed low RIGHT NOW", which has
        # two opposite problems: (1) the pre-race countdown (car
        # naturally stationary for the first ~50 steps of every episode)
        # eats into that same budget, so a car that's merely slow to get
        # going could get unfairly cut off right as it's starting; (2) a
        # car briefly slowing to navigate a tight corner at any point
        # after the threshold would get wrongly flagged as "stuck" even
        # though it's driving normally, since the check only looks at
        # one instant, not whether it's ACTUALLY been stuck for a
        # sustained stretch. Now tracks a consecutive low-speed counter
        # that resets to 0 the moment speed recovers above the
        # threshold -- so the countdown doesn't count against the
        # budget as long as the car accelerates normally afterward, and
        # a momentary slow corner doesn't wrongly trigger this either.
        if obs.get("speedX", 0.0) < ep["stuck_speed_threshold"]:
            self.low_speed_counter += 1
            if self.low_speed_counter > ep["stuck_time_limit"]:
                return True, False, "stuck"
        else:
            self.low_speed_counter = 0

        # --- Lap completed ---
        # Detection: distFromStart jumps from a large value back near 0
        # when the car crosses the start/finish line.
        #
        # IMPORTANT SAFETY CHECK: the car's spawn point can sit just a few
        # meters before the finish line (confirmed from logged data: cars
        # were spawning at distFromStart=6351.65 and reaching the finish
        # line after only ~4m of distRaced). Without a minimum-distance
        # guard, this falsely triggers "lap_complete" almost immediately
        # after every reset, handing out the +50 bonus for doing nothing.
        # We therefore require distRaced to exceed a conservative threshold
        # before accepting the jump as a genuine lap completion.
        dist = obs.get("distFromStart", 0.0)
        dist_raced = obs.get("distRaced", 0.0)
        if self.prev_dist_from_start is not None:
            delta = dist - self.prev_dist_from_start
            # A large negative jump means the car crossed the finish line.
            # Threshold -1000 avoids false triggers from small fluctuations.
            if delta < -1000:
                self.prev_dist_from_start = dist
                if dist_raced >= ep["min_dist_for_lap"]:
                    return True, False, "lap_complete"
                else:
                    # Spawn-point artifact, not a real lap. Treat it as a
                    # normal continuing step -- do NOT terminate, do NOT
                    # award the bonus.
                    pass
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
            "distFromStart", "distRaced", "damage",
            # Actions taken by the agent
            "steer", "accel", "brake", "gear",
            # Reward breakdown (for ablation comparison)
            "r_speed", "r_safety", "r_smooth", "r_anticipation", "r_progress", "r_pedal_conflict", "r_time", "r_terminal", "r_checkpoint", "r_record", "reward_total",
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
            raw_obs.get("damage", 0),
            float(action[0]),
            float(action[1]),
            float(action[2]),
            self.client.R.d.get("gear", 0),
            reward_info.get("r_speed", 0),
            reward_info.get("r_safety", 0),
            reward_info.get("r_smooth", 0),
            reward_info.get("r_anticipation", 0),
            reward_info.get("r_progress", 0),
            reward_info.get("r_pedal_conflict", 0),
            reward_info.get("r_time", 0),
            reward_info.get("terminal", 0),
            reward_info.get("checkpoint", 0),
            reward_info.get("record", 0),
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
    #  Fault-tolerant communication with TORCS
    # ============================================================
    #
    # IMPORTANT: the original snakeoil3 Client.get_servers_input() has an
    # infinite `while True` loop that retries forever if no data arrives.
    # If TORCS crashes, that call would hang the whole training run
    # forever -- it never raises an exception, it just sits there silently.
    # The methods below replace it with a BOUNDED retry that gives up
    # after a fixed number of timeouts and raises TorcsCrashError instead,
    # so the training script can catch it and recover.

    def _safe_get_input(self, max_retries=10):
        """
        Bounded-retry replacement for client.get_servers_input().

        Each retry waits up to 1 second (the socket timeout set by
        snakeoil3). After `max_retries` consecutive timeouts with no
        data at all, we conclude TORCS has crashed and raise
        TorcsCrashError instead of hanging forever.
        """
        sock = self.client.so
        for attempt in range(max_retries):
            try:
                sockdata, _addr = sock.recvfrom(2 ** 17)
                sockdata = sockdata.decode("utf-8")
            except OSError:
                # Timeout or socket error -- TORCS may be slow or dead.
                continue

            if not sockdata:
                continue
            if "***identified***" in sockdata:
                continue  # Handshake echo, keep waiting for real data.
            if "***shutdown***" in sockdata or "***restart***" in sockdata:
                # TORCS itself ended the race -- not a crash, just a
                # normal end-of-race signal. Treat as a clean termination.
                raise TorcsCrashError(
                    "TORCS sent a shutdown/restart signal mid-step. "
                    "This usually means the race ended unexpectedly."
                )

            # Got real sensor data -- success.
            self.client.S.parse_server_str(sockdata)
            return

        # Exhausted all retries with no usable data: TORCS is unresponsive.
        raise TorcsCrashError(
            f"No response from TORCS after {max_retries} attempts "
            f"(~{max_retries} seconds). TORCS has likely crashed."
        )

    def _safe_respond(self):
        """
        Bounded version of client.respond_to_server(). Sending is UDP
        (fire-and-forget), so this rarely blocks, but we still guard it
        in case the socket itself was closed unexpectedly.
        """
        try:
            self.client.respond_to_server()
        except OSError as e:
            raise TorcsCrashError(f"Failed to send action to TORCS: {e}")

    def _wait_for_manual_restart(self, error):
        """
        Pause execution and ask the human to manually restart TORCS.
        Called whenever a TorcsCrashError is raised.
        """
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
        """Send the meta=1 signal telling TORCS to restart the race."""
        if self.client is None:
            return
        try:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()
        except Exception:
            pass  # Connection may already be dead -- fine, reset() will reconnect.
        time.sleep(0.5)  # Give TORCS a moment to process the restart.

    def _disconnect(self):
        """Close the UDP socket if one is open."""
        if self.client is None:
            return
        try:
            self.client.shutdown()
        except Exception:
            pass
        self.client = None

    # ============================================================
    #  Cleanup (called when training ends)
    # ============================================================

    def close(self):
        """Release all resources."""
        # If a log file is still open, this episode was cut short by the
        # training run ending (not a real terminal condition). Write a
        # marker row so the CSV is self-explanatory instead of silently
        # truncating mid-episode (this caused confusion earlier when a
        # log file just stopped with no terminal_reason at all).
        if self._log_writer is not None:
            self._log_writer.writerow(
                [self.time_step] + [""] * 22 + ["training_stopped"]
            )
        self._close_episode_log()
        self._disconnect()
        super().close()