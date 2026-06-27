"""
config.py
=========
All tunable parameters in one place.
For ablation experiments, only change numbers in this file --
no need to touch any logic in gym_torcs_env.py or training scripts.

Usage example (in experiment scripts):
    import config
    config.REWARD_WEIGHTS["w_anticipation"] = 0.3
"""

# ===================== Experiment Tag =====================
# A short, human-readable label for "which version of the reward
# logic/track/etc this batch of experiments used." sweep.py
# automatically embeds this into every result filename and run folder
# name, so you no longer need to invent a descriptive name by hand each
# time you archive a sweep_results.csv or logs folder -- just copy
# this tag into the archived filename, and check EXPERIMENT_LOG.md for
# the full context.
#
# UPDATE THIS whenever you change the reward formula, the track, or
# anything else that would make results incomparable to previous runs.
EXPERIMENT_TAG = "anticipation-v2-sqrt-cone50"

# ===================== Reward Weights =====================
REWARD_WEIGHTS = {
    "w_speed":   1.0,   # Weight for forward progress reward
    "w_safety":  1.0,   # Weight for track-edge / opponent proximity penalty
    "w_smooth":  0.3,   # Weight for steering-smoothness penalty
    # Sole forward-safety mechanism (TTC removed -- single-car-only
    # scope confirmed with supervisor, TTC's car-to-car use case
    # doesn't apply). Default left at the grid-search value currently
    # being tested; sweep.py overrides this per-run regardless.
    "w_anticipation": 0.15,
}

# ===================== Reward Detail Parameters =====================
REWARD_PARAMS = {
    # Tightened from 0.8: the car was "wall-riding" right up to 80% of
    # track width with almost no penalty. Starting the penalty earlier
    # (at 60% of track width) gives the car more warning before it's
    # actually near the edge.
    "safety_margin": 0.6,
    # Safety penalty is now squared (see _compute_safety_reward): this
    # makes the penalty grow much faster as the car gets closer to the
    # edge, instead of growing at a constant rate. A car wall-riding at
    # trackPos=0.95 now gets penalized far more harshly than one merely
    # touching the margin at trackPos=0.65.
    "lateral_penalty_k": 0.5,   # Coefficient penalizing lateral snaking in R_speed
    # Speed x trackPos deviation term, added after reviewing TORCS-RL
    # literature (Wang, Jia & Weng 2018, arXiv:1811.11329 -- TORCS+DDPG,
    # designs own reward to "stick to center of road"; similar
    # speed-scaled deviation terms also appear in other racing-RL
    # papers). The existing squared safety penalty above only depends
    # on position (same penalty whether drifting at 20 km/h or 150
    # km/h at the same trackPos); this term scales the deviation
    # penalty by current speed, so high-speed drifting is penalized
    # more than slow drifting -- complementary to (not redundant
    # with) the existing position-only penalty and the angle-based
    # lateral term already in _compute_speed_reward (that one penalizes
    # heading-angle mismatch, not positional offset).
    # NOTE: exact coefficient from the cited paper not independently
    # verified -- this starting value (0.005) was sized so a typical
    # high-speed deviation (speed=70, trackPos=0.5) produces a modest
    # penalty (~0.18) that doesn't yet dominate R_speed; treat as a
    # starting point to observe and adjust, not a literature-exact value.
    "safety_speed_beta": 0.005,
    # Steering smoothness: penalizes large frame-to-frame changes in the
    # steering action. This directly targets the "jerky driving" and
    # "oversteers into the wall while correcting" behavior observed in
    # training -- right now nothing discourages slamming the wheel from
    # one extreme to the other.
    "smoothness_k": 1.0,
    "time_penalty": -1.0,
    "out_of_track_penalty": -50.0,
    "stuck_penalty": -50.0,
    "backward_penalty": -50.0,
    # New: detected via the 'damage' sensor rising between steps, which is
    # a direct physics-engine signal independent of trackPos geometry.
    # This was added after log analysis showed a high-speed impact
    # (92.7 -> 58.5 km/h in a single step) that trackPos never flagged as
    # out-of-track at this track's geometry -- the car kept dragging
    # itself for ~100 more steps with no termination signal at all.
    "collision_penalty": -50.0,
    # ===== Anticipation reward parameters =====
    # The track[] sensors span -90deg to +90deg in 10deg steps (index 0
    # to 18); index 9 is straight ahead.
    #
    # Reverted from the full 0-18 (-90/+90 deg) sweep back to 4-14
    # (-50/+50 deg): the wider sweep was tested and did NOT improve
    # results -- out_of_track rose from 66.8% to 75.2% overall, and
    # the specific hairpin it was meant to help (Hairpin 2,
    # ~1200-1350m) got WORSE under it (only 22.2% passed safely vs
    # needing further diagnosis). Likely cause: wide-angle side
    # sensors triggered unnecessary slowdowns on ordinary track
    # sections, creating new instability rather than helping at the
    # one hairpin it targeted. Going back to the narrower, previously
    # working range while other variables (ent_coef, smoothness_k)
    # are investigated instead.
    "anticipation_sensor_indices": list(range(4, 15)),
    # Switched from a linear mapping (safe_speed = distance * k) to a
    # sqrt mapping, based on centripetal force physics: the maximum
    # speed a car can take a corner of a given radius without sliding
    # scales with sqrt(radius), not linearly with it (v_max ~
    # sqrt(mu * g * r)). The old linear formula was too permissive at
    # larger clearances, allowing high speed well before a corner
    # actually required slowing down.
    "anticipation_speed_per_sqrt_meter": 12.0,  # km/h per sqrt(meter) of forward clearance
    "anticipation_max_safe_speed": 200.0,  # Cap so long straights don't get penalized
    "lap_complete_bonus": 50.0,  # Will be recalibrated once laps are reliably completed
}

# ===================== Episode Termination Parameters =====================
EPISODE_PARAMS = {
    "stuck_speed_threshold": 5,   # km/h -- below this for too long = stuck
    "stuck_time_limit": 500,      # Steps before generic stuck detection activates
    "max_steps": 10000,           # Hard cap per episode
    # Minimum distRaced (meters) required before a distFromStart jump is
    # accepted as a real lap completion. Without this, a car that spawns
    # close to the finish line can falsely trigger "lap_complete" after
    # driving only a few meters. Set conservatively low for now; raise
    # this once we know the track's real length from logged data.
    "min_dist_for_lap": 1000,
    # "Wall-pinned" detection: a MUCH faster timeout than the generic
    # stuck check above, specifically for the case where the car is
    # jammed right up against the track edge with almost no speed.
    # Observed behavior: the car wall-rides, stalls completely, then
    # burns through the full 500-step stuck timer before restarting --
    # wasting hundreds of steps in a situation with no recovery value.
    # This check ends those episodes much sooner so training time isn't
    # wasted on dead situations. (True active recovery -- reversing and
    # countersteering out -- is deferred to the future rule-based
    # behavior-cloning teacher script, NOT implemented here, because
    # overriding the agent's chosen action during PPO training would
    # corrupt its credit assignment: the logged action wouldn't match
    # what TORCS actually executed.)
    "wall_pinned_trackpos_threshold": 0.8,
    "wall_pinned_speed_threshold": 3,
    "wall_pinned_time_limit": 100,
    # Reverse-assist: once the car has been wall-pinned for this many
    # steps (well before the 100-step termination above), gear is
    # switched to reverse. This does NOT override the agent's chosen
    # steer/accel values -- it only changes which gear they're applied
    # through, so the agent still experiences the true consequences of
    # its own action choices (gear is already rule-based, same as
    # always; this is not a new credit-assignment risk). The agent gets
    # 70 steps (100-30) to discover that reversing+steering away works,
    # before the episode would otherwise be terminated.
    "reverse_assist_time_limit": 30,
}

# ===================== Traction Control =====================
# Mirrors the simple TCL logic from the original rule-based driver
# (torcs_jm_par.py's traction_control()): if the rear wheels are
# spinning noticeably faster than the front ones, the tires are slipping
# rather than gripping, so accel is dampened. This is a deterministic,
# physics-level safety filter applied to the actuator -- not a
# "strategic" decision -- so it's handled the same way as gear shifting:
# outside the PPO action space, applied to the action right before
# sending it to TORCS.
TRACTION_CONTROL = {
    "enabled": True,
    "wheel_spin_diff_threshold": 2.0,  # Triggers when rear-front spin diff exceeds this
    "accel_reduction": 0.2,            # Amount subtracted from accel when slipping
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

# ===================== Observation/Reward Normalization =====================
# Logged training data showed value_loss climbing steadily while
# explained_variance stayed near zero across multiple runs -- a classic
# symptom of un-normalized rewards/observations making it hard for the
# Critic network to learn a stable value scale. VecNormalize (applied in
# train.py) tracks a running mean/std and rescales obs and rewards on the
# fly, without needing to hand-pick scale constants ourselves.
NORMALIZE = {
    "enabled": True,
    "norm_obs": True,
    "norm_reward": True,
    "clip_obs": 10.0,
    "clip_reward": 10.0,
    "stats_path": "./checkpoints/vecnormalize_stats.pkl",
}

# ===================== PPO Training Hyperparameters =====================
# These are passed directly into stable-baselines3's PPO constructor.
# Defaults below are reasonable starting points for continuous-control
# tasks; we will tune them after seeing the first training results.
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,          # Steps collected per policy update
    "batch_size": 64,
    "n_epochs": 10,           # Optimization passes per update
    "gamma": 0.99,            # Discount factor
    "gae_lambda": 0.95,       # Advantage estimation smoothing
    "clip_range": 0.2,        # PPO's signature clipping parameter
    # Raised from 0.0: logged training data showed std (action spread)
    # shrinking from ~0.97 to ~0.76 and entropy_loss rising toward zero
    # across runs -- the policy was converging prematurely, which lines
    # up with the observed left-wall-riding habit becoming persistent.
    # A small entropy bonus keeps some exploration alive so the policy
    # has a chance to escape habits like this instead of locking in.
    "ent_coef": 0.01,
    "verbose": 1,
}

# ===================== Training Run Settings =====================
TRAINING = {
    "total_timesteps": 400_000,   # Overall training budget for this run
    "checkpoint_freq": 5_000,     # Save a checkpoint every N timesteps
    "checkpoint_dir": "./checkpoints",
    "model_save_path": "./checkpoints/torcs_ppo_final",
    "tensorboard_log_dir": "./tb_logs",
}