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


# ===================== Reward Weights =====================
REWARD_WEIGHTS = {
    # LOWERED further from 0.4 to 0.15, backed by REAL measured data
    # (check_reward_composition.py on 20 real episodes): speed reward
    # was 126% of total reconstructed reward -- meaning without it, the
    # total would be NEGATIVE (all other components combined were net
    # negative). Speed was single-handedly propping up the entire
    # reward signal, while checkpoint+record combined were under 1%.
    # This isn't a guess anymore, it's measured: speed needs to come
    # down sharply for the milestone/record rewards (raised below) to
    # have any real chance of competing.
    "w_speed":   0.3,
    "w_safety":  1.0,   # Weight for track-edge / opponent proximity penalty
    "w_smooth":  0.3,   # Weight for steering-smoothness penalty
    "w_anticipation": 0.25,
    "w_progress": 1.0,  # Weight for the potential-based progress-shaping term (see _compute_progress_reward)
    "w_pedal_conflict": 1.0,  # Weight for the accel/brake-overlap penalty (see _compute_pedal_conflict_reward)
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
    # ===== Pedal conflict penalty (see _compute_pedal_conflict_reward) =====
    # Added after direct log evidence of accel and brake both being
    # pressed hard at the same step, repeatedly, while the car barely
    # moved. threshold=0.3 per the original request ("both > 0.3");
    # k=10 sized so a FULL overlap (accel=1.0,brake=1.0, min=1.0) gives
    # a clearly noticeable single-step penalty (-10), deliberately
    # larger than smoothness's max contribution (~0.6 weighted) since
    # this is currently the most direct, evidence-backed behavioral fix
    # -- not yet validated against real retraining data, watch
    # check_reward_composition.py after the next run to see its actual
    # share of total reward.
    "pedal_conflict_threshold": 0.3,
    "pedal_conflict_k": 10.0,
    "time_penalty": -1.0,
    # RAISED (all four terminal penalties below, roughly 6-8x): backed
    # by REAL measured data (check_reward_composition.py) -- across 20
    # episodes, the average per-episode terminal penalty was only -2.7
    # (most episodes didn't even crash; the few that did paid -50,
    # diluted by non-crash episodes in the average). Even a full -50
    # single-instance hit is small relative to typical per-episode
    # speed-reward accumulation (hundreds), even with w_speed already
    # lowered to 0.15. Collision/out_of_track (genuinely dangerous,
    # avoidable failures) are penalized more heavily than stuck/backward
    # (more "gave up" than "crashed") to preserve some differentiation.
    "out_of_track_penalty": -600.0,
    "stuck_penalty": -450.0,
    "backward_penalty": -450.0,
    # RAISED 1.5x again (alongside the checkpoint increase above) to
    # preserve roughly the same proportional safety margin -- the
    # largest single checkpoint (750 at 3400m) still slightly exceeds
    # this (600), leaving a small residual "rush the last checkpoint
    # then crash" profit margin (+150, down from what would have been
    # +600 under a flat 2x checkpoint scaling with unchanged penalties)
    # -- not fully eliminated, but reaching 3400m at all already
    # represents enormous progress, so the practical risk this
    # encourages recklessness specifically there is low.
    "collision_penalty": -600.0,
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
    # Raised from 150 to comfortably exceed the new checkpoint sum below
    # (270) -- must stay clearly larger so completing the FULL lap is
    # still worth more than the sum of all the sub-goals along the way.
    # RAISED again from 350: the checkpoint bonuses above were just
    # scaled up 10x (new sum = 2830), so this must scale up too to stay
    # ahead of that sum and preserve the intended ordering (full lap >
    # sum of all sub-goals along the way).
    "lap_complete_bonus": 6000.0,
    # ===== Checkpoint rewards (one-time per checkpoint, per episode) =====
    # See _compute_checkpoint_reward in gym_torcs_env.py for full
    # rationale. EXPANDED from 3 to 9 checkpoints, spaced roughly every
    # 400m, covering most of the known Corkscrew lap length (3608.45m),
    # stopping at 3400m (leaving the final stretch to the finish line as
    # genuinely distinct from "just another checkpoint" -- that's what
    # lap_complete_bonus rewards). The near checkpoints (200/600m) are
    # within or just beyond the car's current actual reach (~200-220m
    # after 114 episodes) so they function as real near-term incentives
    # right now; the farther ones (1000m+) are aspirational for now and
    # will start mattering once progress moves past the near ones --
    # left in place rather than added later, so the reward structure
    # doesn't need another revision once the car improves.
    # Bonuses increase with distance (10 -> 50) to reflect that farther
    # checkpoints represent more accumulated progress/difficulty. Sum =
    # 270, comfortably under the new lap_complete_bonus of 350.
    # Added 100m/150m after the fact: these are even closer to the car's
    # current actual reach than 200m, giving it the most immediate,
    # easiest-to-hit incentives right at the very start of a run. Sum is
    # now 283 (was 270), still comfortably under lap_complete_bonus=350.
    # SCALED UP 10x from the previous values, backed by REAL measured
    # data (check_reward_composition.py): checkpoint+record combined
    # were under 1% of total reward across 20 real episodes --
    # essentially invisible to the policy. Sum is now 2830 (was 283).
    # SCALED 1.5x (not 2x) from the previous values, per explicit
    # request to raise these further -- but capped at 1.5x rather than
    # the originally-considered 2x because doubling would have let the
    # largest single checkpoint (3400m) exceed the collision penalty
    # outright (1000 vs 400), reopening the "rush to a far checkpoint
    # then crash, still net profit" loophole this project fixed
    # elsewhere before. Terminal penalties below are raised 1.5x in
    # lockstep to preserve roughly the same proportional safety margin.
    # New sum = 4245 (was 2830).
    "checkpoints": [
        (100.0,  75.0),
        (150.0,  120.0),
        (200.0,  150.0),
        (600.0,  225.0),
        (1000.0, 300.0),
        (1400.0, 375.0),
        (1800.0, 450.0),
        (2200.0, 525.0),
        (2600.0, 600.0),
        (3000.0, 675.0),
        (3400.0, 750.0),
    ],
    # ===== Progress-shaping coefficients (see _compute_progress_reward) =====
    # Potential function: Phi = distRaced - lateral_k*|trackPos| - angle_k*|angle|.
    # These weights are NOT independently derived/tuned -- they're a
    # reasonable starting point (position deviation penalized more
    # heavily than heading-angle deviation, consistent with trackPos
    # being the more direct out_of_track risk signal) and should be
    # revisited once you see how this term behaves in practice.
    "progress_lateral_k": 100.0,
    "progress_angle_k": 30.0,
    # ===== Record-breaking reward (one-time, persists across episodes) =====
    # Rewards the car for reaching a NEW farthest distance EVER seen in
    # this training run -- not per-episode, a single global high-water
    # mark. Added because the user's priority is now explicitly "just
    # finish a lap, speed doesn't matter" -- this directly rewards
    # exploring further than the policy has ever gotten before,
    # regardless of how it got there or how fast.
    # Safe by construction against the "flicker/oscillate to farm
    # reward" exploit class found elsewhere in this project: the
    # high-water mark only ever increases, so the SAME ground can never
    # pay out twice.
    # min_improvement: a new max must beat the old one by at least this
    # many meters to count -- without this, tiny random fluctuations of
    # a few cm past the old record would constantly trigger trivial,
    # noisy payouts with no real significance.
    "record_min_improvement": 10.0,
    # Reward scales with sqrt(new record distance) rather than linearly
    # with it -- linear scaling would make later record-breaks grow
    # explosively large (e.g. 10x the distance = 10x the reward), while
    # sqrt scaling still rewards later breakthroughs MORE than early
    # ones (matching "later progress should be worth more"), just less
    # explosively, keeping magnitudes from overwhelming everything else
    # as distances grow into the thousands of meters.
    # RAISED from 5.0 to 50.0, backed by REAL measured data
    # (check_reward_composition.py): record+checkpoint combined were
    # under 1% of total reward. At k=50, breaking a record at 500m now
    # gives 50*sqrt(500)~=1118 (was ~112) -- a magnitude that can
    # actually compete with speed's measured contribution.
    "record_reward_k": 50.0,
}

# ===================== Episode Termination Parameters =====================
EPISODE_PARAMS = {
    "stuck_speed_threshold": 5,   # km/h -- below this for too long = stuck
    # LOWERED from 1000: diagnosed as a major contributor to the car
    # learning to just stop early instead of attempting the track --
    # RAISED from 150 to 800: diagnosed root cause of early "stuck"
    # episodes is steering oscillation (steer flipping between extremes
    # almost every step), not an unwillingness to move -- accel was
    # often near max, but chaotic steering meant it never translated
    # into real forward progress (0.1m over 150 steps in one observed
    # episode). 800 gives random exploration more time to stumble onto
    # an effective steer+accel combo before being cut off. This does
    # NOT fix the steering-chaos issue itself (that's a training-time
    # problem, not a config value) -- it just buys more random-search
    # time per attempt. Accepted tradeoff: more real wall-clock time
    # spent on early, likely-still-ineffective episodes.
    "stuck_time_limit": 800,      # Steps before generic stuck detection activates
    # RAISED from 5000: physical calculation shows this was an
    # impossible ceiling for completing a full lap regardless of
    # driving quality. Corkscrew is 3608.45m; even at a modest 50 km/h
    # average (the user has explicitly deprioritized speed, so this
    # should be achievable), completing a lap takes ~3608/13.9 m/s =
    # ~260s, which at the observed 26-40 fps in this project's training
    # logs is roughly 6700-10400 steps -- already exceeding 5000 before
    # accounting for any slower sections (corners, recovery). 15000
    # gives real headroom for a full lap even at a leisurely pace.
    "max_steps": 15000,           # Hard cap per episode
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
    # Grace period before out_of_track actually terminates (see
    # _check_terminal in gym_torcs_env.py). Includes an anti-flicker
    # safeguard from the start: the car must stay continuously within
    # bounds for out_of_track_recovery_streak_required steps before the
    # off-track counter clears, so it can't "flicker" across the
    # boundary to avoid ever accumulating enough off-track steps to
    # terminate while still effectively riding the edge.
    "out_of_track_grace_steps": 80,
    "out_of_track_recovery_streak_required": 15,
    "wall_pinned_trackpos_threshold": 0.8,
    "wall_pinned_speed_threshold": 3,
    "wall_pinned_time_limit": 1000,
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
    # DISABLED: norm_reward's running variance estimate can be
    # permanently skewed by rare, very-high-total-reward episodes (e.g.
    # the 300k+/2500m outliers seen in training -- whether from the new
    # checkpoint/lap bonuses or simply natural variance in how far an
    # episode gets). clip_reward only bounds the OUTPUT after division,
    # it does NOT protect the underlying running-std estimate itself,
    # which is computed from the raw, unclipped reward. Once an outlier
    # episode inflates that running std, every subsequent NORMAL
    # episode's reward gets divided by an artificially large number,
    # suppressing/distorting the effective learning signal for a long
    # stretch afterward -- a plausible mechanism for the repeated
    # "rises well, then catastrophically and permanently collapses"
    # pattern seen across multiple different reward designs in this
    # project. Turning this off trades away automatic reward scaling
    # for removing this instability source; reward magnitudes need to
    # be kept in a sane manual range instead (this project has mostly
    # been doing that anyway via REWARD_PARAMS/REWARD_WEIGHTS tuning).
    "norm_reward": False,
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
    "batch_size": 128,
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
    # NEW: caps how much a single update is allowed to change the
    # policy (measured via approx_kl), stopping the remaining
    # mini-batch updates in that rollout early if exceeded. Added after
    # repeated, unresolved "rises well then collapses permanently"
    # episodes in this project, on the hypothesis that a stretch of bad
    # data (e.g. many consecutive collisions) could occasionally drive
    # one update far enough to overwrite an already-good policy. Sized
    # against recently observed approx_kl values (consistently
    # ~0.008-0.009 during normal training) -- 0.02 gives headroom for
    # ordinary updates to pass through unaffected, while still capping
    # any future update that's roughly 2x+ larger than that normal
    # range, which is the actual failure mode this is meant to guard
    # against.
    "target_kl": 0.02,
    "verbose": 1,
}

# ===================== Resume-only overrides =====================
# These are NOT valid PPO() constructor kwargs -- they are read
# explicitly by train.py's --resume path (see main()) to override a
# LOADED model's hyperparameters, since PPO.load() restores whatever
# was saved at checkpoint time, not whatever is in PPO_PARAMS above.
# Diagnosed need: training repeatedly showed std (action exploration
# spread) climbing throughout an entire ~1.7M-step run and never
# coming back down, alongside a catastrophic, permanent collapse in
# distance/reward partway through that several different reward
# designs in this project have now shown -- pointing at a training-
# dynamics issue (runaway exploration + a constant, never-decayed
# learning rate late into training) rather than purely a reward-design
# issue.
RESUME_OVERRIDES = {
    # One-time step-down (not a smooth decay -- see the comment in
    # train.py for why ent_coef can't be scheduled like learning_rate
    # natively in SB3). Lowered from the original 0.01.
    "ent_coef": 0.003,
    # Linear decay from the ORIGINAL learning_rate down to this value,
    # spread across the remaining total_timesteps of the resumed run.
    "lr_decay_to": 5e-5,
}

# ===================== Training Run Settings =====================
TRAINING = {
    "total_timesteps": 2_000_000,   # Overall training budget for this run
    "checkpoint_freq": 5_000,     # Save a checkpoint every N timesteps
    "checkpoint_dir": "./checkpoints",
    "model_save_path": "./checkpoints/torcs_ppo_final",
    "tensorboard_log_dir": "./tb_logs",
}