# Sector SAC Residual RL

This is the reinforcement-learning path for improving lap time without giving up the stable rule/profile driver.

The agent does not directly steer the car. It acts once per 50m sector and, by default, learns one residual:

- target speed multiplier delta

The high-speed brake target residual is disabled by default. The baseline remains `configs/speed_profile_v2_12_finish_brake_170.json`, which has already finished one lap at about `125.404s`. SAC is rewarded by beating the baseline sector-by-sector, not by a vague per-frame reward.

The main speed reward is normalized sector-time improvement:

```text
(reference_sector_steps - actual_sector_steps) / reference_sector_steps
```

For the same fixed 50m sector, using fewer frames means the car crossed that sector faster. Raw progress reward is disabled by default because the baseline already completes the lap; the task is now to keep completion while reducing time.

Sectors `44-50` and `61-66` are locked to the baseline residual action by default because the first continuous SAC runs repeatedly failed after downhill/approach sections into sectors `49` and `65`. The environment also applies temporary entry caps in those windows:

- `44-45`: speed multiplier capped at `1.05`
- `46-50`: speed multiplier capped at `1.00`
- `61-62`: speed multiplier capped at `1.05`
- `63-66`: speed multiplier capped at `1.00`

These caps are training scaffolding, not the final racing strategy. They keep the best baseline stable while SAC learns speed residuals on safer sectors. Once completion is stable, loosen the caps gradually, for example `61-62` from `1.05` to `1.08`, then `44-45` from `1.05` to `1.08`.

## 1. Build or refresh the reference lap

The repo already contains `configs/sector_reference_v2_12_50m.json`, generated from:

```powershell
python build_sector_reference.py --telemetry runs\20260630_050228_profile_v2_12_finish_brake_170\telemetry_episode_0.csv --out configs\sector_reference_v2_12_50m.json --reference-lap-time 125.404
```

If you produce a better deterministic profile later, regenerate this file from that run's telemetry.

## 2. Train SAC

Start TORCS with `scr_server` waiting on port `3001`, then run:

```powershell
python train_sector_sac.py --out-dir models\sector_sac_speed_only --total-timesteps 50000
```

The first conservative run uses:

- speed delta: `+/- 0.04`
- brake target delta: disabled
- locked sectors: `44-50,61-66`
- max speed multiplier: `1.44`
- max brake target speed: `170 km/h`

This keeps exploration close to the known-finish baseline while still allowing full `+/-0.04` speed residual on ordinary sectors. Do not enable brake residuals until the model finishes several laps reliably.

Offtrack/backward failures always receive a base failure penalty. High-speed offtrack receives an extra speed-scaled penalty, and near-edge/large-heading states receive early per-sector penalties before the car actually leaves the track. This keeps the speed reward from teaching the agent to trade stability for a short burst of speed.

## 3. Evaluate a trained policy

```powershell
python race_sector_policy_driver.py --model models\sector_sac_speed_only\final_model.zip --episodes 3
```

Current best stable checkpoint:

```powershell
python race_sector_policy_driver.py --model models\best\sector_sac_best_stable.zip --episodes 3
```

This model is copied from `models\sector_sac_straightboost_v1\checkpoints\sector_sac_1000_steps.zip`. Later training in that run stayed complete but became slower, so the early checkpoint is preferred over `final_model.zip`.

Acceptance checks:

- `reason=target_laps` on every episode
- final `frame_count` below the reference `5939`
- lap time below about `125.4s`

During the capped fine-tune stage, first check stability rather than raw speed:

- `sector_entry_cap` appears for sectors `44-50` and `61-66`
- `unsafe_fraction` stays near `0.0` in sectors `49` and `65`
- the run finishes several laps before caps are relaxed

Only after that should you try a wider range:

```powershell
python train_sector_sac.py --out-dir models\sector_sac_v2 --resume-from models\sector_sac_speed_only\final_model.zip --total-timesteps 50000 --speed-delta 0.04 --enable-brake-delta --brake-delta-kmh 3 --max-speed-multiplier 1.44 --max-brake-target-speed 173
```

## Why this should be more learnable

The earlier residual PPO made thousands of tiny frame-level decisions per lap. A failure at a corner made it hard to identify which earlier action caused the problem.

This environment makes about 73 decisions per lap. Reward is tied to the exact sector where time was gained or lost, while the rule controller still keeps the car on a stable line.
