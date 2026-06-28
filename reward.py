"""
reward.py
=========
Pure-function reward computation for the TORCS PPO agent.

Design contract
---------------
Every function in this module is a PURE FUNCTION with respect to its
inputs: it reads `obs` (raw sensor dict) and explicit scalar arguments,
consults config constants, and returns a float.  No function mutates any
object it receives.

The one unavoidable "state across steps" requirement (smoothness penalty
needs the previous steer value; checkpoint/record rewards need episode and
run-level accumulators) is handled via two explicit dataclasses:

    EpisodeRewardState  -- reset to fresh_episode_state() each episode
    RunRewardState      -- created once per training run, never reset

TorcsEnv owns both objects and passes them explicitly to the functions
that need them.  After each step, TorcsEnv replaces the EpisodeRewardState
with the updated copy returned by compute_reward().  This makes every
state transition visible at the call site, eliminates hidden side-effects
inside reward methods, and lets check_reward_composition.py replay reward
calculations against CSV data without instantiating TorcsEnv at all.

Usage (inside TorcsEnv.step)
-----------------------------
    reward, info, self.ep_reward_state = compute_reward(
        obs=raw_obs,
        steer=steer,
        accel=accel,
        brake=brake,
        ep_state=self.ep_reward_state,
        run_state=self.run_reward_state,
        terminal_reason=terminal_reason,
    )
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

import config


# ================================================================
#  State dataclasses (owned by TorcsEnv, passed in explicitly)
# ================================================================

@dataclass
class EpisodeRewardState:
    """
    Mutable reward-related state that resets every episode.

    Attributes
    ----------
    prev_steer : float or None
        Steer value from the previous step; used by the smoothness penalty.
        None at the start of each episode (first step has no delta).
    checkpoints_reached : set[float]
        Distance thresholds (meters) that have already paid out their
        one-time bonus this episode.  Cleared in fresh_episode_state().
    """
    prev_steer: float | None = None
    checkpoints_reached: Set[float] = field(default_factory=set)


@dataclass
class RunRewardState:
    """
    Reward-related state that persists across episodes for the entire
    training run.

    Attributes
    ----------
    furthest_distance_ever : float
        Global high-water mark for distRaced (meters).  Only ever
        increases.  NOT saved/restored with model checkpoints -- a
        --resume will restart this from 0 (see config.py comment for
        the known implication).
    """
    furthest_distance_ever: float = 0.0


def fresh_episode_state() -> EpisodeRewardState:
    """Return a clean EpisodeRewardState for the start of a new episode."""
    return EpisodeRewardState()


# ================================================================
#  Per-step reward components (stateless pure functions)
# ================================================================

def compute_core_progress(obs: dict) -> float:
    """
    R_core_progress = speedX*cos(angle)
                      - speedX*|sin(angle)| * lateral_penalty_k
                      - speedX*|trackPos|   * core_progress_beta

    Modeled on a working reference PPO-for-TORCS implementation.  All
    three terms share speedX as a factor, so the whole thing collapses
    toward zero as speed approaches zero -- no speed-independent
    component that can fire while the car is stationary.
    """
    speed_x   = obs.get("speedX", 0.0)
    angle     = obs.get("angle", 0.0)
    track_pos = obs.get("trackPos", 0.0)
    p = config.REWARD_PARAMS

    forward        = speed_x * math.cos(angle)
    lateral        = speed_x * abs(math.sin(angle)) * p["lateral_penalty_k"]
    position_cross = speed_x * abs(track_pos)       * p["core_progress_beta"]

    return forward - lateral - position_cross


def compute_safety(obs: dict) -> float:
    """
    R_safety = -max(0, |trackPos| - safety_margin)^2

    Speed-INDEPENDENT by design: riding the wall should be penalized even
    at zero speed (e.g. a car stopped against a barrier).  Grows quadratically
    as the car approaches the edge beyond the margin.
    """
    track_pos = abs(obs.get("trackPos", 0.0))
    margin    = config.REWARD_PARAMS["safety_margin"]

    if track_pos > margin:
        return -((track_pos - margin) ** 2)
    return 0.0


def compute_smoothness(steer: float, prev_steer: float | None) -> float:
    """
    R_smooth = -smoothness_k * |steer - prev_steer|

    Penalizes large frame-to-frame steering changes.  Returns 0.0 on the
    first step of an episode (prev_steer is None).

    NOTE: this function is pure -- it does NOT update prev_steer.
    The caller (compute_reward) returns the new prev_steer value in the
    updated EpisodeRewardState.
    """
    if prev_steer is None:
        return 0.0
    delta = abs(steer - prev_steer)
    return -config.REWARD_PARAMS["smoothness_k"] * delta


def compute_anticipation(obs: dict) -> float:
    """
    Penalizes maintaining high speed when forward-facing sensors show the
    track narrowing ahead (approaching corner).

    Uses a sqrt relationship (safe_speed = k * sqrt(distance)) grounded in
    centripetal-force physics: max cornering speed scales with sqrt(radius).
    """
    p = config.REWARD_PARAMS
    track   = obs.get("track", [200.0] * 19)
    indices = p["anticipation_sensor_indices"]
    sensors = [track[i] for i in indices if i < len(track)]
    forward_distance = max(min(sensors) if sensors else 200.0, 0.0)

    safe_speed = min(
        p["anticipation_speed_per_sqrt_meter"] * math.sqrt(forward_distance),
        p["anticipation_max_safe_speed"],
    )

    excess = max(0.0, obs.get("speedX", 0.0) - safe_speed)
    return -excess


# ================================================================
#  One-time reward components
# ================================================================

def get_terminal_reward(reason: str) -> float:
    """
    One-shot reward/penalty at the moment the episode terminates.
    Pure function: reads config, returns float, no side-effects.
    """
    p = config.REWARD_PARAMS
    mapping = {
        "collision":    p["collision_penalty"],
        "out_of_track": p["out_of_track_penalty"],
        "stuck":        p["stuck_penalty"],
        "wall_pinned":  p["stuck_penalty"],
        "backward":     p["backward_penalty"],
        "lap_complete": p["lap_complete_bonus"],
    }
    return mapping.get(reason, 0.0)


def compute_checkpoint_reward(
    obs: dict,
    terminal_reason: str,
    ep_state: EpisodeRewardState,
) -> Tuple[float, EpisodeRewardState]:
    """
    One-time bonus the first time distRaced crosses each configured
    checkpoint distance within an episode.

    Returns
    -------
    bonus : float
        Sum of bonuses for newly crossed checkpoints this step.
    new_ep_state : EpisodeRewardState
        Updated state with the newly reached checkpoints recorded.
        The CALLER must store this; this function does not mutate ep_state.
    """
    if terminal_reason in ("collision", "out_of_track"):
        return 0.0, ep_state

    dist_raced       = obs.get("distRaced", 0.0)
    total_bonus      = 0.0
    new_reached      = set(ep_state.checkpoints_reached)

    for threshold, bonus in config.REWARD_PARAMS["checkpoints"]:
        if threshold in new_reached:
            continue
        if dist_raced >= threshold:
            new_reached.add(threshold)
            total_bonus += bonus

    new_state = EpisodeRewardState(
        prev_steer=ep_state.prev_steer,
        checkpoints_reached=new_reached,
    )
    return total_bonus, new_state


def compute_record_reward(
    obs: dict,
    terminal_reason: str,
    run_state: RunRewardState,
) -> Tuple[float, RunRewardState]:
    """
    One-time bonus when distRaced reaches a new all-time high for this
    training run.

    Returns
    -------
    bonus : float
    new_run_state : RunRewardState
        Updated state with furthest_distance_ever advanced if a record
        was broken.  The CALLER must store this.
    """
    if terminal_reason in ("collision", "out_of_track"):
        return 0.0, run_state

    p          = config.REWARD_PARAMS
    dist_raced = obs.get("distRaced", 0.0)

    if dist_raced < run_state.furthest_distance_ever + p["record_min_improvement"]:
        return 0.0, run_state

    new_run_state = RunRewardState(furthest_distance_ever=dist_raced)
    bonus         = p["record_reward_k"] * (dist_raced ** 0.5)
    return bonus, new_run_state


# ================================================================
#  Orchestrator
# ================================================================

def compute_reward(
    obs: dict,
    steer: float,
    accel: float,
    brake: float,
    ep_state: EpisodeRewardState,
    run_state: RunRewardState,
    terminal_reason: str,
) -> Tuple[float, Dict[str, float], EpisodeRewardState, RunRewardState]:
    """
    Compute the full composite reward for one step.

    R_total = w_core_progress * R_core_progress
            + w_safety        * R_safety
            + w_smooth        * R_smooth
            + w_anticipation  * R_anticipation
            + time_penalty
            + R_terminal  (one-time, non-zero only on terminating steps)
            + R_checkpoint (one-time, non-zero when crossing a milestone)
            + R_record     (one-time, non-zero on new furthest-ever distance)

    Parameters
    ----------
    obs : dict
        Raw sensor dict from TORCS (client.S.d).
    steer : float
        Clipped steer value sent to TORCS this step.
    accel : float
        Clipped (and traction-controlled) accel value sent this step.
        Currently unused in reward computation but accepted for forward-
        compatibility -- add accel/brake-based reward terms here without
        changing the call site.
    brake : float
        Clipped brake value sent to TORCS this step.  Same note as accel.
    ep_state : EpisodeRewardState
        Episode-level reward accumulator (prev_steer, checkpoints_reached).
    run_state : RunRewardState
        Run-level reward accumulator (furthest_distance_ever).
    terminal_reason : str
        Termination reason string from termination.check_terminal(), or ""
        if the episode continues.

    Returns
    -------
    total : float
    info  : dict[str, float]   -- individual components for CSV logging
    new_ep_state  : EpisodeRewardState  -- caller must replace ep_state with this
    new_run_state : RunRewardState      -- caller must replace run_state with this
    """
    w = config.REWARD_WEIGHTS

    r_core_progress = compute_core_progress(obs)
    r_safety        = compute_safety(obs)
    r_smooth        = compute_smoothness(steer, ep_state.prev_steer)
    r_anticipation  = compute_anticipation(obs)
    r_time          = config.REWARD_PARAMS["time_penalty"]
    r_terminal      = get_terminal_reward(terminal_reason)

    r_checkpoint, ep_state_after_ckpt = compute_checkpoint_reward(
        obs, terminal_reason, ep_state
    )
    r_record, new_run_state = compute_record_reward(
        obs, terminal_reason, run_state
    )

    total = (
        w["w_core_progress"] * r_core_progress
        + w["w_safety"]      * r_safety
        + w["w_smooth"]      * r_smooth
        + w["w_anticipation"] * r_anticipation
        + r_time
        + r_terminal
        + r_checkpoint
        + r_record
    )

    info = {
        "r_core_progress": r_core_progress,
        "r_safety":        r_safety,
        "r_smooth":        r_smooth,
        "r_anticipation":  r_anticipation,
        "r_time":          r_time,
        "terminal":        r_terminal,
        "checkpoint":      r_checkpoint,
        "record":          r_record,
    }

    # Build the updated episode state: advance prev_steer, carry forward
    # the updated checkpoints_reached from compute_checkpoint_reward.
    new_ep_state = EpisodeRewardState(
        prev_steer=steer,
        checkpoints_reached=ep_state_after_ckpt.checkpoints_reached,
    )

    return total, info, new_ep_state, new_run_state
