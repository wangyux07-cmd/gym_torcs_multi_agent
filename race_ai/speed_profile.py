from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .controller import clip


@dataclass(frozen=True)
class SpeedSegment:
    start: float
    end: float
    multiplier: float
    name: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SpeedSegment":
        return cls(
            start=float(payload["start"]),
            end=float(payload["end"]),
            multiplier=float(payload["multiplier"]),
            name=str(payload.get("name", "")),
        )

    def contains(self, distance: float, lap_length: float) -> bool:
        start = self.start % lap_length
        end = self.end % lap_length
        distance = distance % lap_length
        if start <= end:
            return start <= distance < end
        return distance >= start or distance < end


@dataclass
class SpeedProfile:
    lap_length: float = 3608.45
    default_multiplier: float = 1.0
    min_multiplier: float = 0.92
    max_multiplier: float = 1.12
    caution_multiplier: float = 0.98
    segments: tuple[SpeedSegment, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SpeedProfile":
        return cls(
            lap_length=float(payload.get("lap_length", 3608.45)),
            default_multiplier=float(payload.get("default_multiplier", 1.0)),
            min_multiplier=float(payload.get("min_multiplier", 0.92)),
            max_multiplier=float(payload.get("max_multiplier", 1.12)),
            caution_multiplier=float(payload.get("caution_multiplier", 0.98)),
            segments=tuple(SpeedSegment.from_mapping(item) for item in payload.get("segments", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lap_length": self.lap_length,
            "default_multiplier": self.default_multiplier,
            "min_multiplier": self.min_multiplier,
            "max_multiplier": self.max_multiplier,
            "caution_multiplier": self.caution_multiplier,
            "segments": [
                {"start": item.start, "end": item.end, "multiplier": item.multiplier, "name": item.name}
                for item in self.segments
            ],
        }

    def raw_multiplier(self, distance: float) -> float:
        for segment in self.segments:
            if segment.contains(distance, self.lap_length):
                return clip(segment.multiplier, self.min_multiplier, self.max_multiplier)
        return clip(self.default_multiplier, self.min_multiplier, self.max_multiplier)

    def safe_multiplier(self, sensors: Mapping[str, Any], rule_steer: float) -> float:
        distance = float(sensors.get("distFromStart", sensors.get("distRaced", 0.0))) % self.lap_length
        desired = self.raw_multiplier(distance)

        track = list(sensors.get("track", [200.0] * 19))
        ahead = float(track[9])
        side_clearance = min(float(value) for value in track[4:15])
        angle_abs = abs(float(sensors.get("angle", 0.0)))
        track_pos_abs = abs(float(sensors.get("trackPos", 0.0)))
        steer_abs = abs(rule_steer)

        caution_cap = max(self.min_multiplier, min(1.0, self.caution_multiplier))
        if angle_abs > 0.55 or track_pos_abs > 0.72:
            return min(desired, caution_cap)
        if ahead < 45.0 or side_clearance < 14.0 or steer_abs > 0.28 or angle_abs > 0.36 or track_pos_abs > 0.55:
            return min(desired, 1.0)
        if ahead < 75.0 or side_clearance < 24.0 or steer_abs > 0.18 or angle_abs > 0.24 or track_pos_abs > 0.38:
            return min(desired, 1.04)
        return desired


def load_speed_profile(path: str | Path) -> SpeedProfile:
    with Path(path).open("r", encoding="utf-8") as fh:
        return SpeedProfile.from_mapping(json.load(fh))


def save_speed_profile(profile: SpeedProfile, path: str | Path) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as fh:
        json.dump(profile.to_dict(), fh, indent=2, sort_keys=True)
