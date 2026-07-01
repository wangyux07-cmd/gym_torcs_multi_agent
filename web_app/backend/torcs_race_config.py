from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


DEFAULT_TRACKS = [
    {"category": "road", "name": "corkscrew", "label": "Corkscrew"},
    {"category": "road", "name": "aalborg", "label": "Aalborg"},
    {"category": "road", "name": "g-track-1", "label": "G-Track 1"},
    {"category": "road", "name": "wheel-2", "label": "Wheel 2"},
    {"category": "oval", "name": "michigan", "label": "Michigan"},
]


def normalize_laps(value: Any) -> int:
    try:
        laps = int(value)
    except (TypeError, ValueError):
        return 1
    return min(20, max(1, laps))


def torcs_root_from_exe(torcs_path: str) -> Path:
    exe = Path(torcs_path)
    if exe.name.lower() in {"wtorcs.exe", "torcs.exe"}:
        return exe.parent
    return exe


def discover_tracks(torcs_path: str) -> list[dict[str, str]]:
    root = torcs_root_from_exe(torcs_path) if torcs_path else Path("D:/torcs")
    tracks_root = root / "tracks"
    if not tracks_root.exists():
        return list(DEFAULT_TRACKS)

    tracks: list[dict[str, str]] = []
    for category in ("road", "oval", "dirt"):
        category_dir = tracks_root / category
        if not category_dir.exists():
            continue
        for track_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            if (track_dir / f"{track_dir.name}.xml").exists():
                tracks.append(
                    {
                        "category": category,
                        "name": track_dir.name,
                        "label": format_track_label(track_dir.name),
                    }
                )
    return tracks or list(DEFAULT_TRACKS)


def format_track_label(name: str) -> str:
    return " ".join(part.capitalize() for part in name.replace("_", "-").split("-"))


def validate_track(settings: dict[str, Any], tracks: list[dict[str, str]]) -> tuple[str, str]:
    category = str(settings.get("track_category", "road")).strip() or "road"
    name = str(settings.get("track_name", "corkscrew")).strip() or "corkscrew"
    for track in tracks:
        if track["category"] == category and track["name"] == name:
            return category, name
    fallback = tracks[0] if tracks else DEFAULT_TRACKS[0]
    return fallback["category"], fallback["name"]


def write_practice_config(settings: dict[str, Any]) -> Path:
    torcs_path = str(settings.get("torcs_path", ""))
    if not torcs_path:
        raise FileNotFoundError("Set the simulator path in Garage first.")
    root = torcs_root_from_exe(torcs_path)
    if not root.exists():
        raise FileNotFoundError(f"Simulator folder does not exist: {root}")

    tracks = discover_tracks(torcs_path)
    category, track_name = validate_track(settings, tracks)
    laps = normalize_laps(settings.get("laps"))
    config_path = root / "config" / "raceman" / "practice.xml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_practice_config(category, track_name, laps), encoding="utf-8")
    return config_path


def render_practice_config(category: str, track_name: str, laps: int) -> str:
    category = escape(category)
    track_name = escape(track_name)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "params.dtd">


<params name="Practice">
  <section name="Header">
    <attstr name="name" val="Practice"/>
    <attstr name="description" val="Practice"/>
    <attnum name="priority" val="100"/>
    <attstr name="menu image" val="data/img/splash-practice.png"/>
    <attstr name="run image" val="data/img/splash-run-practice.png"/>
  </section>

  <section name="Tracks">
    <attnum name="maximum number" val="1"/>
    <section name="1">
      <attstr name="name" val="{track_name}"/>
      <attstr name="category" val="{category}"/>
    </section>
  </section>

  <section name="Races">
    <section name="1">
      <attstr name="name" val="Practice"/>
    </section>
  </section>

  <section name="Practice">
    <attnum name="laps" val="{laps}"/>
    <attstr name="type" val="practice"/>
    <attstr name="starting order" val="drivers list"/>
    <attstr name="restart" val="yes"/>
    <attstr name="display mode" val="normal"/>
    <attstr name="display results" val="yes"/>
    <attnum name="distance" unit="km" val="0"/>
    <section name="Starting Grid">
      <attnum name="rows" val="1"/>
      <attnum name="distance to start" val="100"/>
      <attnum name="distance between columns" val="20"/>
      <attnum name="offset within a column" val="10"/>
      <attnum name="initial speed" unit="km/h" val="0"/>
      <attnum name="initial height" unit="m" val="0.2"/>
    </section>
  </section>

  <section name="Drivers">
    <attnum name="maximum number" val="1"/>
    <attnum name="focused idx" val="0"/>
    <attstr name="focused module" val="scr_server"/>
    <section name="1">
      <attnum name="idx" val="0"/>
      <attstr name="module" val="scr_server"/>
    </section>
  </section>

  <section name="Configuration">
    <attnum name="current configuration" val="4"/>
    <section name="1">
      <attstr name="type" val="track select"/>
    </section>
    <section name="2">
      <attstr name="type" val="drivers select"/>
    </section>
    <section name="3">
      <attstr name="type" val="race config"/>
      <attstr name="race" val="Practice"/>
      <section name="Options">
        <section name="1">
          <attstr name="type" val="race length"/>
        </section>
        <section name="2">
          <attstr name="type" val="display mode"/>
        </section>
      </section>
    </section>
  </section>

  <section name="Drivers Start List">
    <section name="1">
      <attstr name="module" val="scr_server"/>
      <attnum name="idx" val="0"/>
    </section>
  </section>
</params>
'''
