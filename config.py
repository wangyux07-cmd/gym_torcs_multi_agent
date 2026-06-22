"""
config.py
=========
All tunable parameters in one place.
For ablation experiments, only change numbers in this file --
no need to touch any logic in gym_torcs_env.py or training scripts.

Usage example (in experiment scripts):
    import config
    config.REWARD_WEIGHTS["w_ttc"] = 0.3   # enable TTC penalty
"""

# ===================== Reward Weights =====================
# Single-car stage: w_ttc stays at 0 (disabled).
# Multi-car stage: set w_ttc to a non-zero value to enable it.
REWARD_WEIGHTS = {
    "w_speed":  1.0,   # Weight for forward progress reward
    "w_safety": 0.5,   # Weight for track-edge / opponent proximity penalty
    "w_ttc":    0.0,   # Weight for Time-to-Collision penalty (0 = off for single-car)
}

# ===================== Reward Detail Parameters =====================
REWARD_PARAMS = {
    "safety_margin": 0.8,       # Safety penalty kicks in when |trackPos| exceeds this
    "lateral_penalty_k": 0.5,   # Coefficient penalizing lateral snaking in R_speed
    "out_of_track_penalty": -10.0,
    "stuck_penalty": -10.0,
    "backward_penalty": -10.0,
    "lap_complete_bonus": 50.0, # Will be recalibrated after seeing actual reward scale
}

# ===================== Episode Termination Parameters =====================
EPISODE_PARAMS = {
    "stuck_speed_threshold": 5,   # km/h -- below this for too long = stuck
    "stuck_time_limit": 500,      # Steps before stuck detection activates
    "max_steps": 10000,           # Hard cap per episode
}

# ===================== TORCS Connection =====================
TORCS_PARAMS = {
    "port": 3001,
    "host": "localhost",
}

# ===================== State Dimensions =====================
# angle(1) + trackPos(1) + speedX(1) + speedY(1) + speedZ(1)
# + track(19) + wheelSpinVel(4) + rpm(1) + opponents(36) = 65
STATE_DIM = 65

# ===================== Action Dimensions =====================
# steer [-1,1], accel [0,1], brake [0,1]
ACTION_DIM = 3

# ===================== Logging =====================
LOGGING = {
    "enabled": True,       # Set False to skip CSV logging during heavy training
    "log_dir": "./logs",   # Directory for per-episode CSV files
}