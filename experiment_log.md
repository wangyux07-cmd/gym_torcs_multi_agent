# Experiment Log

This file is the single source of truth for "which archived file/folder
corresponds to which experiment." Whenever you rename a `sweep_results.csv`
or `sweep_logs` / `logs` folder to archive it, add a row here on the SAME
day, while you still remember the context. Don't rely on memory later --
write it down immediately after each run.

| Archive name | Date | Track | Reward formula version | Parameters swept | Steps/run | Notes |
|---|---|---|---|---|---|---|
| `logs_old_track1_with_ttc` | ~2026-06 (mixed sessions) | Old track (track1) | Had TTC + old anticipation (linear, ±30° cone) | N/A -- accumulated debugging history, not a single clean run | Mixed | Contains multiple training sessions interleaved; not reliable for thesis conclusions. See chat history for the "why this is unreliable" diagnosis. |
| `sweep_results_30k.csv` + `sweep_logs_30k` | ~2026-06 | New track (Corkscrew) | Anticipation only (linear formula, ±30° cone), TTC already removed | `w_anticipation` (0.15, 0.3, 0.6); `anticipation_speed_per_meter` (0.6, 1.0, 1.5) | 30,000 | First exploratory one-at-a-time sweep. Too few steps -- avg_speed only 9-18 km/h, results are mostly noise. Use only the GROUPED AVERAGE ablation comparison (anticipation on vs off), not individual cell values. |
| `sweep_results_old.csv` + `sweep_logs_old` | ~2026-06 | New track (Corkscrew) | Anticipation only (linear formula, ±30° cone) | `w_anticipation` x `w_safety` grid, FIRST grid search attempt | 30,000 | First grid search, before the column-alignment bug in sweep.py was fixed. Terminal-reason columns may be misaligned -- only trust ep_len_mean/ep_rew_mean from this file. |
| `sweep_results_old2.csv` | ~2026-06 | New track (Corkscrew) | Anticipation only (linear formula, ±30° cone) | First grid search attempt, before the sweep.py column-alignment bug was fixed | 30,000 | First grid search, with the unreliable terminal-reason columns (only trust ep_len_mean/ep_rew_mean from this file). |
| `sweep_results_oldformula.csv` + `sweep_logs_oldformula` | ~2026-06 | New track (Corkscrew) | Anticipation only (linear formula, ±30° cone) | `w_anticipation=0.15` x `w_safety` (1.5, 2.0) -- the SECOND, correctly-formatted grid search attempt | 200,000 | **This is the important one**: completed cells (0.15, 1.5) and (0.15, 2.0). Diagnostic finding: raising w_safety made out_of_track WORSE, not better (161 -> 179 failures) despite higher avg speed -- this directly motivated the anticipation redesign (wider cone, sqrt formula). Keep this file -- it's the "before" evidence for your methods narrative. |
| *(next: new sweep with redesigned anticipation)* | | New track (Corkscrew) | Anticipation v2: sqrt formula + ±50° cone | `w_anticipation` (0.15, 0.3) x `w_safety` (1.5, 2.0, 2.5) | 200,000 | This is the "after" evidence -- compare against `sweep_results_oldformula.csv` to show the redesign's effect. Will use config.EXPERIMENT_TAG="anticipation-v2-sqrt-cone50" automatically once you run the updated sweep.py. |

## Naming convention going forward

When you archive a folder, use this format so the name carries the key
facts without needing to open this file first:

```
sweep_results_<YYYYMMDD>_<formula-version-tag>.csv
sweep_logs_<YYYYMMDD>_<formula-version-tag>/
```

Example: `sweep_results_20260625_anticipation-v2-sqrt.csv`

The "formula version tag" should match `config.EXPERIMENT_TAG` (see the
config.py change below) -- the script now embeds this automatically into
new run folder names, so you mostly just need to copy that same tag into
the archived file's name when you rename it.