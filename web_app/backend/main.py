from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .database import init_db
from .race_runner import RaceRunner
from .results_store import ResultsStore
from .settings_store import SettingsStore
from .simulator import SimulatorManager, has_torcs_process, navigate_torcs_to_race, mjpeg_stream
from .torcs_race_config import discover_tracks, write_practice_config


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web_app"
FRONTEND = WEB_ROOT / "frontend"
DATA = WEB_ROOT / "data"
DB_PATH = DATA / "race.db"

init_db(DB_PATH)

app = FastAPI(title="AI Race Control")
results = ResultsStore(DB_PATH)
settings = SettingsStore(DB_PATH)
runner = RaceRunner(ROOT, results)
simulator = SimulatorManager(settings)

app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def get_user_id(request: Request) -> str:
    uid = request.headers.get("X-User-Id", "").strip().lower()
    if _UUID_RE.match(uid):
        return uid
    return "default"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/status")
def status() -> dict:
    return runner.snapshot()


@app.get("/api/tracks")
def list_tracks(user_id: str = Depends(get_user_id)) -> dict:
    s = settings.load(user_id)
    return {"tracks": discover_tracks(s.get("torcs_path", ""))}


@app.post("/api/race/start")
def start_race(
    user_id: str = Depends(get_user_id),
    payload: Optional[dict] = Body(default=None),
) -> dict:
    try:
        s = settings.load(user_id)
        if payload:
            for key in ("track_category", "track_name", "laps", "driver_config"):
                if key in payload:
                    s[key] = str(payload[key]).strip()
        settings.save(user_id, s)

        torcs_path = s.get("torcs_path", "")
        if not torcs_path:
            raise FileNotFoundError("Set the simulator path in Garage first.")
        if not has_torcs_process(torcs_path):
            raise FileNotFoundError("TORCS is not running. Click Launch TORCS first.")
        write_practice_config(s)
        navigate_torcs_to_race(torcs_path, s.get("window_title", "TORCS"))
        time.sleep(3.0)

        target_laps = max(1, int(s.get("laps", 1)))
        config_path = s.get("driver_config", "configs/rule_fast.json")
        username = results.get_username(user_id)
        return runner.start(user_id=user_id, username=username, target_laps=target_laps, config_path=config_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/race/stop")
def stop_race() -> dict:
    return runner.stop()


@app.get("/api/profile")
def get_profile(user_id: str = Depends(get_user_id)) -> dict:
    return {"user_id": user_id, "username": results.get_username(user_id)}


@app.post("/api/profile")
def save_profile(
    user_id: str = Depends(get_user_id),
    payload: dict = Body(),
) -> dict:
    username = results.set_username(user_id, str(payload.get("username", "")))
    return {"user_id": user_id, "username": username}


@app.get("/api/results")
def list_results(user_id: str = Depends(get_user_id)) -> dict:
    return {"results": results.list_results(user_id)}


@app.get("/api/results/best")
def best_result(user_id: str = Depends(get_user_id)) -> dict:
    return {"best": results.best_result(user_id)}


@app.get("/api/leaderboard")
def leaderboard() -> dict:
    return {"leaderboard": results.leaderboard()}


@app.get("/api/garage")
def garage(user_id: str = Depends(get_user_id)) -> dict:
    best = results.best_result(user_id)
    config = runner.driver_config
    setup_name = Path(config).stem.replace("_", " ").title()
    return {
        "driver_name": "AI Driver",
        "setup_name": setup_name,
        "status": "Ready",
        "reliability": "Stable",
        "driver_config": config,
        "best_lap": None if best is None else best.get("lap_time"),
    }


@app.get("/api/settings")
def get_settings(user_id: str = Depends(get_user_id)) -> dict:
    return settings.load(user_id)


@app.post("/api/settings")
def save_settings(
    user_id: str = Depends(get_user_id),
    payload: dict = Body(),
) -> dict:
    return settings.save(user_id, payload)


@app.get("/api/simulator/status")
def simulator_status(user_id: str = Depends(get_user_id)) -> dict:
    return simulator.status(user_id)


@app.post("/api/simulator/launch")
def launch_simulator(user_id: str = Depends(get_user_id)) -> dict:
    try:
        s = settings.load(user_id)
        write_practice_config(s)
        return simulator.launch(user_id=user_id, force_restart=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/simulator/stop")
def stop_simulator(user_id: str = Depends(get_user_id)) -> dict:
    return simulator.stop(user_id)


@app.get("/api/simulator/view.mjpg")
def simulator_view() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream(simulator),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
