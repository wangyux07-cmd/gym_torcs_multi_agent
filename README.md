# RaceYourCode — AI TORCS Driver

An AI racing platform built on [TORCS](https://torcs.sourceforge.net/). A rule-based controller provides a stable, fast baseline; residual RL layers (PPO and SAC) learn to push lap times further without sacrificing stability.

A web interface lets multiple users race the AI, track per-driver results, and compare lap times on a shared leaderboard.

---

## Quick Start

### Web Interface

```bash
pip install -r requirements-web.txt
python run_web_app.py
```

Open **http://127.0.0.1:8000**, enter your name in Garage, set the TORCS simulator path, then click **Launch TORCS → Race Now**.

### Command Line

```bash
# Start TORCS and wait for the blue "Waiting for driver" screen, then:
python race_rule_driver.py --target-laps 1 --episodes 1
```

---

## Requirements

| Component | Notes |
|-----------|-------|
| Python 3.10+ | |
| TORCS with SCR server (`wtorcs.exe`) | Windows build with `scr_server` driver module |
| `requirements-web.txt` | Web interface (FastAPI, uvicorn, mss, pywin32) |
| `requirements-rl.txt` | RL training and inference (stable-baselines3, gymnasium, torch) |

---

## Project Structure

```
gym_torcs/
├── race_ai/
│   ├── controller.py            Rule-based driver (PID steering, heuristic throttle/brake)
│   ├── residual_env.py          Gymnasium env for PPO residual training
│   ├── sector_residual_env.py   Gymnasium env for SAC sector-level training
│   ├── speed_profile.py         Distance-indexed speed multiplier profiles
│   ├── sector_reference.py      Reference lap split into 50 m sectors
│   ├── metrics.py               Episode telemetry writer (CSV + JSONL)
│   ├── train_residual.py        PPO training entry point
│   ├── train_sector_sac.py      SAC training entry point
│   └── tune.py                  Optuna hyperparameter search for the rule controller
├── web_app/
│   ├── backend/                 FastAPI REST API + TORCS process management
│   └── frontend/                Single-page app (vanilla JS, no build step)
├── configs/
│   ├── rule_fast.json           Default rule controller parameters
│   ├── rule_tuned.json          Optuna-tuned variant
│   ├── speed_profile_*.json     Distance-based speed multipliers
│   └── sector_reference_*.json  Reference lap sector data
├── models/                      Trained PPO and SAC checkpoints
├── race_rule_driver.py          CLI: rule-based driver
├── race_profile_driver.py       CLI: rule driver + speed profile
├── race_residual_driver.py      CLI: rule driver + PPO residual correction
├── race_sector_policy_driver.py CLI: rule driver + SAC sector policy
├── build_sector_reference.py    Build sector reference JSON from a telemetry CSV
├── eval_residual_env_baseline.py  Validate the env with zero RL action (sanity check)
├── snakeoil3_gym.py             TORCS UDP client (Chris X Edwards)
└── run_web_app.py               Launch web server
```

---

## Drivers

### Rule-based (`race_rule_driver.py`)

PID steering with look-ahead track sensors, heuristic throttle, blind-corner braking, and rear-wheel traction control. All tunable parameters live in `configs/rule_fast.json`.

```bash
python race_rule_driver.py --config configs/rule_fast.json --target-laps 1
```

### Speed Profile (`race_profile_driver.py`)

Rule driver with a distance-indexed speed multiplier map. Specific track sections can run faster or slower than the base rule speed.

```bash
python race_profile_driver.py --profile configs/speed_profile_v3.json --telemetry
```

### Residual PPO (`race_residual_driver.py`)

A trained PPO policy outputs a speed multiplier that replaces the rule controller's target speed. Steering and braking remain rule-based; safety caps prevent the policy from exceeding safe limits.

```bash
python race_residual_driver.py \
  --model models/residual_ppo/final_model.zip \
  --action-mode speed_target
```

### Sector SAC (`race_sector_policy_driver.py`)

SAC acts once per 50 m track sector and adjusts a small speed offset on top of the speed profile. High-risk sectors can be locked so the policy cannot modify them.

```bash
python race_sector_policy_driver.py \
  --model models/best/sector_sac_best_stable.zip \
  --locked-sectors 44-50,61-66
```

---

## Training

### Tune the rule controller (Optuna)

```bash
python -m race_ai.tune --trials 30 --best-config-out configs/rule_tuned.json
```

### Train residual PPO

```bash
python -m race_ai.train_residual --total-timesteps 200000 --out-dir models/my_ppo
```

### Train sector SAC

```bash
python -m race_ai.train_sector_sac --total-timesteps 500000 --out-dir models/my_sac
```

---

## Configuration

| File | What it controls |
|------|-----------------|
| `configs/rule_fast.json` | kp/kd gains, target speeds, brake thresholds, gear shift points |
| `configs/speed_profile_*.json` | Per-segment multipliers; `default_multiplier` applies off-segment |
| `configs/sector_reference_*.json` | Reference steps per 50 m sector; drives SAC reward shaping |

### Building a sector reference

Collect a telemetry lap first, then convert it:

```bash
python race_profile_driver.py --telemetry --episodes 1 --target-laps 1

python build_sector_reference.py \
  --telemetry runs/<run_id>/telemetry_episode_0.csv \
  --out configs/sector_reference_new.json
```

---

## Acknowledgements

- `snakeoil3_gym.py` — TORCS UDP client adapted from [Chris X Edwards](http://xed.ch/project/snakeoil/).
- Simulator: [TORCS](https://torcs.sourceforge.net/) with SCR server extensions.
- Original gym wrapper: [gym_torcs](https://github.com/ugo-nama-kun/gym_torcs) (Preferred Networks, 2016).
