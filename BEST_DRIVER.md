# Best AI Driver

The current default AI driver for demos and the GUI is:

```text
models/best/sector_sac_best_stable.zip
```

This checkpoint was copied from:

```text
models/sector_sac_straightboost_v1/checkpoints/sector_sac_1000_steps.zip
```

Selection reason:

- `sector_sac_entrycap_v2` reached the fastest observed single lap, about `123.320s`, but later had sector `65` offtrack failures.
- `sector_sac_straightboost_v1` at the early checkpoint was more suitable for delivery: the first 10 completed laps finished without failure, with best lap about `123.718s` and mean lap about `124.693s`.
- Later `straightboost_v1` training degraded, so the final model is not used as the default.

The GUI should hide algorithm details and launch this best stable driver by default.
