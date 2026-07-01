# Residual RL for TORCS

This project now treats the rule controller as the stable baseline and trains PPO to output only small residual corrections.

The default mode is `speed_target`: the rule controller keeps steering, throttle, brake, and gear authority. PPO can only request a small target-speed multiplier, and safety gates cap that multiplier before tight corners.

The default baseline is `configs/rule_fast.json`. Avoid using `configs/rule_tuned.json` for residual training unless it is proven faster and stable on your track.

## Install dependencies

```powershell
python -m pip install -r requirements-rl.txt
```

If you use a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
```

## Train

Start TORCS with `scr_server` waiting on port `3001`, then run:

```powershell
python -m race_ai.train_residual --base-config configs/rule_fast.json --out-dir models/residual_speed_target --total-timesteps 100000 --action-mode speed_target --boost-start-distance 650 --boost-deadband 0.25 --max-speed-multiplier 1.08 --multiplier-step-limit 0.01 --caution-speed-multiplier 0.98 --heading-penalty-scale 0.35 --learning-rate 0.00005 --gamma 0.995
```

The residual action range is intentionally conservative. PPO is not learning to drive from zero here; it is learning where the rule driver's target speed can be slightly higher without losing the stable racing line.

## Run the trained model

```powershell
python race_residual_driver.py --config configs/rule_fast.json --model models/residual_speed_target/final_model.zip --action-mode speed_target --episodes 1 --target-laps 1
```

If the trained model makes the baseline worse, reduce the correction strength:

```powershell
python race_residual_driver.py --config configs/rule_fast.json --model models/residual_speed_target/final_model.zip --action-mode speed_target --residual-scale 0.5
```

To confirm the runner itself matches the rule baseline, disable residual corrections:

```powershell
python race_residual_driver.py --config configs/rule_fast.json --model models/residual_speed_target/final_model.zip --action-mode speed_target --residual-scale 0.0
```

## Recommended workflow

1. Verify `race_rule_driver.py --config configs/rule_fast.json` completes laps reliably.
2. Train residual PPO for a short run first, around `100000` timesteps.
3. Compare lap time against the baseline using the same TORCS track and run settings.
4. Only keep the residual model if it improves lap time without increasing crashes.
5. After residual RL is stable, build the GUI around these two modes: rule baseline and rule + residual model.
