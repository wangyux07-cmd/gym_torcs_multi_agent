"""
config.py
=========
All tunable parameters in one place.
For ablation experiments, only change numbers in this file --
no need to touch any logic in gym_torcs_env.py or training scripts.
"""

# ===================== Reward Weights =====================
REWARD_WEIGHTS = {
    "w_speed":        1.0,   # Weight for forward progress reward
    "w_safety":       1.0,   # Weight for track-edge penalty (squared)
    "w_smooth":       0.3,   # Weight for steering-smoothness penalty
    # Penalizes maintaining high speed when front-facing sensors show the
    # track narrowing ahead (approaching corner). Started conservative --
    # this is a deliberate signal, not a dominant one.
    "w_anticipation": 0.3,
}

# ===================== Reward Detail Parameters =====================
REWARD_PARAMS = {
    # --- Safety ---
    # Tightened from 0.8: the car was wall-riding right up to 80% of
    # track width with almost no penalty. Starting earlier (at 60%)
    # gives the car more warning before it's actually near the edge.
    "safety_margin": 0.6,
    "lateral_penalty_k": 0.5,   # Coefficient penalizing lateral snaking in R_speed

    # --- Smoothness ---
    "smoothness_k": 1.0,        # Penalty coefficient for steer delta per step

    # --- Time ---
    "time_penalty": -1.0,       # Flat per-step cost (encourages speed)

    # --- Terminal events ---
    "out_of_track_penalty": -50.0,
    "stuck_penalty":        -50.0,
    "backward_penalty":     -50.0,
    # Detected via the 'damage' sensor rising between steps -- a direct
    # physics-engine signal independent of trackPos geometry.
    "collision_penalty":    -50.0,
    "lap_complete_bonus":    50.0,

    # --- Anticipation ---
    # The track[] sensors span -90deg to +90deg in 10deg steps (index 0
    # to 18); index 9 is straight ahead. Indices 6-12 cover -30deg to
    # +30deg, a forward-facing cone wide enough to catch an approaching
    # corner without being thrown off by sensors pointed too far sideways.
    "anticipation_sensor_indices":   list(range(6, 13)),
    "anticipation_speed_per_meter":  1.0,    # km/h of safe speed per meter of clearance
    "anticipation_max_safe_speed":   200.0,  # Cap so long straights are never penalized
}

# ===================== Episode Termination Parameters =====================
EPISODE_PARAMS = {
    "stuck_speed_threshold": 5,   # km/h -- below this for too long = stuck
    "stuck_time_limit": 500,      # Steps before generic stuck detection activates
    "max_steps": 10000,
    "min_dist_for_lap": 1000,     # meters, guards against spawn-point false lap
    "wall_pinned_trackpos_threshold": 0.8,
    "wall_pinned_speed_threshold": 3,
    "wall_pinned_time_limit": 100,
    "reverse_assist_time_limit": 30,
}

# ===================== Traction Control =====================
TRACTION_CONTROL = {
    "enabled": True,
    "wheel_spin_diff_threshold": 2.0,
    "accel_reduction": 0.2,
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
ACTION_DIM = 3

# ===================== Logging =====================
LOGGING = {
    "enabled": True,
    "log_dir": "./logs",
}

# ===================== Observation/Reward Normalization =====================
NORMALIZE = {
    "enabled": True,
    "norm_obs": True,
    "norm_reward": True,
    "clip_obs": 10.0,
    "clip_reward": 10.0,
    "stats_path": "./checkpoints/vecnormalize_stats.pkl",
}

# ===================== PPO Training Hyperparameters =====================
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "verbose": 1,
}

# ===================== Training Run Settings =====================
TRAINING = {
    "total_timesteps": 200_000,
    "checkpoint_freq": 5_000,
    "checkpoint_dir": "./checkpoints",
    "model_save_path": "./checkpoints/torcs_ppo_final",
    "tensorboard_log_dir": "./tb_logs",
}