# AI Race Control Web App

This web app is the non-technical control surface for the TORCS AI driver.

It does not expose algorithm choices. The default race button launches the best stable AI driver:

```text
models/best/sector_sac_best_stable.zip
```

## Run

Start the web app:

```powershell
python -m pip install -r requirements-web.txt
python run_web_app.py
```

Open:

```text
http://127.0.0.1:8000
```

In the Garage panel, save the path to `torcs.exe`, then click `Launch Simulator`.
Keep the TORCS window visible; the web app shows a live preview of that window.

## What It Does

- `Race Now` starts one AI-driven lap.
- `Launch Simulator` opens TORCS from the saved path.
- The TORCS simulator remains a native window, while the web app displays a live preview.
- The browser shows race state, lap progress, result summary, garage status, and leaderboard.
- Results are stored locally in `web_app/data/results.json`.
