"""
termination.py
==============
Pure-function episode termination logic for the TORCS PPO agent.

Design contract
---------------
All termination state (counters that accumulate across steps within a single
episode) lives in the TerminationState dataclass.  TorcsEnv owns this object
and passes it in explicitly; check_terminal() returns a new copy with any
updated counter values.  No function here mutates its inputs.

This separation means:
  - Changing a termination condition only requires editing this file.
  - The change has no risk of accidentally touching reward or logging logic.
  - Unit-testing termination conditions requires no TORCS connection or Gym
    environment -- just call check_terminal() with a hand-crafted obs dict.

Usage (inside TorcsEnv.step)
-----------------------------
    terminated, truncated, reason, self.term_state = check_terminal(
        obs=raw_obs,
        prev_obs=self.prev_obs_raw,
        time_step=self.time_step,
        state=self.term_state,
    )
"""

import math
from dataclasses import dataclass
from typing import Tuple

import config


# ================================================================
#  State dataclass (owned by TorcsEnv, passed in explicitly)
# ================================================================

@dataclass
class TerminationState:
    """
    All counters needed to evaluate termination conditions across steps.

    Every field resets to its default value at the start of each episode
    via fresh_termination_state().

    Attributes
    ----------
    out_of_track_counter : int
        Consecutive steps where |trackPos| > 1.0.  Episode ends when this
        exceeds EPISODE_PARAMS["out_of_track_grace_steps"].
    in_track_streak : int
        Consecutive steps back within bounds.  Must reach
        EPISODE_PARAMS["out_of_track_recovery_streak_required"] before
        out_of_track_counter resets to 0 (anti-flicker guard).
    wall_pinned_counter : int
        Consecutive steps where the car is jammed against the wall with
        near-zero speed.  Episode ends when this exceeds
        EPISODE_PARAMS["wall_pinned_time_limit"].
    low_speed_counter : int
        Consecutive steps below EPISODE_PARAMS["stuck_speed_threshold"].
        Resets to 0 the moment speed recovers.
    prev_dist_from_start : float or None
        distFromStart reading from the previous step; used to detect the
        large negative jump that signals a lap completion.  None on the
        first step of each episode.
    """
    out_of_track_counter:    int   = 0
    in_track_streak:         int   = 0
    wall_pinned_counter:     int   = 0
    low_speed_counter:       int   = 0
    prev_dist_from_start:    float | None = None


def fresh_termination_state(initial_dist_from_start: float | None = None) -> TerminationState:
    """Return a clean TerminationState for the start of a new episode."""
    return TerminationState(prev_dist_from_start=initial_dist_from_start)


# ================================================================
#  Core check
# ================================================================

def check_terminal(
    obs: dict,
    prev_obs: dict | None,
    time_step: int,
    state: TerminationState,
) -> Tuple[bool, bool, str, TerminationState]:
    """
    Evaluate all termination conditions for the current step.

    Parameters
    ----------
    obs : dict
        Current raw sensor dict (client.S.d after the step).
    prev_obs : dict or None
        Raw sensor dict from the PREVIOUS step (for damage-delta check).
        None only on the very first step of the first episode.
    time_step : int
        Steps elapsed in the current episode (0-indexed).
    state : TerminationState
        Counter state carried in from the previous step.

    Returns
    -------
    terminated : bool
        True if the episode ended due to a game-logic condition.
    truncated : bool
        True if the episode ended due to the step limit.
    reason : str
        Human-readable label ("collision", "out_of_track", "backward",
        "wall_pinned", "stuck", "lap_complete", "max_steps", or "" if
        the episode continues).
    new_state : TerminationState
        Updated counter state.  Caller must replace state with this.
    """
    ep = config.EPISODE_PARAMS

    # Work on mutable locals; we will build new_state at the end.
    out_of_track_counter = state.out_of_track_counter
    in_track_streak      = state.in_track_streak
    wall_pinned_counter  = state.wall_pinned_counter
    low_speed_counter    = state.low_speed_counter
    prev_dist_from_start = state.prev_dist_from_start

    # ------------------------------------------------------------------
    # 1. Collision (damage-based, highest priority)
    # ------------------------------------------------------------------
    # TORCS increments 'damage' whenever the car hits something, even at
    # track locations where trackPos never crosses the out-of-track boundary.
    prev_damage = prev_obs.get("damage", 0.0) if prev_obs else 0.0
    if obs.get("damage", 0.0) - prev_damage > 0:
        new_state = TerminationState(
            out_of_track_counter=out_of_track_counter,
            in_track_streak=in_track_streak,
            wall_pinned_counter=wall_pinned_counter,
            low_speed_counter=low_speed_counter,
            prev_dist_from_start=prev_dist_from_start,
        )
        return True, False, "collision", new_state

    # ------------------------------------------------------------------
    # 2. Out of track (grace-period, with anti-flicker streak guard)
    # ------------------------------------------------------------------
    if abs(obs.get("trackPos", 0.0)) > 1.0:
        out_of_track_counter += 1
        in_track_streak = 0
        if out_of_track_counter > ep["out_of_track_grace_steps"]:
            new_state = TerminationState(
                out_of_track_counter=out_of_track_counter,
                in_track_streak=in_track_streak,
                wall_pinned_counter=wall_pinned_counter,
                low_speed_counter=low_speed_counter,
                prev_dist_from_start=prev_dist_from_start,
            )
            return True, False, "out_of_track", new_state
    else:
        in_track_streak += 1
        if in_track_streak >= ep["out_of_track_recovery_streak_required"]:
            out_of_track_counter = 0

    # ------------------------------------------------------------------
    # 3. Running backward
    # ------------------------------------------------------------------
    if math.cos(obs.get("angle", 0.0)) < 0:
        new_state = TerminationState(
            out_of_track_counter=out_of_track_counter,
            in_track_streak=in_track_streak,
            wall_pinned_counter=wall_pinned_counter,
            low_speed_counter=low_speed_counter,
            prev_dist_from_start=prev_dist_from_start,
        )
        return True, False, "backward", new_state

    # ------------------------------------------------------------------
    # 4. Wall-pinned (fast timeout for edge-jammed + near-zero speed)
    # ------------------------------------------------------------------
    track_pos_abs = abs(obs.get("trackPos", 0.0))
    if (track_pos_abs > ep["wall_pinned_trackpos_threshold"]
            and obs.get("speedX", 0.0) < ep["wall_pinned_speed_threshold"]):
        wall_pinned_counter += 1
        if wall_pinned_counter > ep["wall_pinned_time_limit"]:
            new_state = TerminationState(
                out_of_track_counter=out_of_track_counter,
                in_track_streak=in_track_streak,
                wall_pinned_counter=wall_pinned_counter,
                low_speed_counter=low_speed_counter,
                prev_dist_from_start=prev_dist_from_start,
            )
            return True, False, "wall_pinned", new_state
    else:
        wall_pinned_counter = 0

    # ------------------------------------------------------------------
    # 5. Stuck (consecutive low-speed counter)
    # ------------------------------------------------------------------
    if obs.get("speedX", 0.0) < ep["stuck_speed_threshold"]:
        low_speed_counter += 1
        if low_speed_counter > ep["stuck_time_limit"]:
            new_state = TerminationState(
                out_of_track_counter=out_of_track_counter,
                in_track_streak=in_track_streak,
                wall_pinned_counter=wall_pinned_counter,
                low_speed_counter=low_speed_counter,
                prev_dist_from_start=prev_dist_from_start,
            )
            return True, False, "stuck", new_state
    else:
        low_speed_counter = 0

    # ------------------------------------------------------------------
    # 6. Lap completed (large negative jump in distFromStart)
    # ------------------------------------------------------------------
    dist            = obs.get("distFromStart", 0.0)
    dist_raced      = obs.get("distRaced", 0.0)
    lap_complete    = False

    if prev_dist_from_start is not None:
        delta = dist - prev_dist_from_start
        if delta < -1000 and dist_raced >= ep["min_dist_for_lap"]:
            lap_complete = True

    prev_dist_from_start = dist  # always advance for next step

    if lap_complete:
        new_state = TerminationState(
            out_of_track_counter=out_of_track_counter,
            in_track_streak=in_track_streak,
            wall_pinned_counter=wall_pinned_counter,
            low_speed_counter=low_speed_counter,
            prev_dist_from_start=prev_dist_from_start,
        )
        return True, False, "lap_complete", new_state

    # ------------------------------------------------------------------
    # 7. Step limit
    # ------------------------------------------------------------------
    new_state = TerminationState(
        out_of_track_counter=out_of_track_counter,
        in_track_streak=in_track_streak,
        wall_pinned_counter=wall_pinned_counter,
        low_speed_counter=low_speed_counter,
        prev_dist_from_start=prev_dist_from_start,
    )

    if time_step >= ep["max_steps"]:
        return False, True, "max_steps", new_state

    # Episode continues.
    return False, False, "", new_state
