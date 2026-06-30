from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SectorTarget:
    index: int
    start: float
    end: float
    reference_steps: int
    reference_speed: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SectorTarget":
        return cls(
            index=int(payload["index"]),
            start=float(payload["start"]),
            end=float(payload["end"]),
            reference_steps=int(payload["reference_steps"]),
            reference_speed=float(payload.get("reference_speed", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "reference_steps": self.reference_steps,
            "reference_speed": self.reference_speed,
        }


@dataclass(frozen=True)
class SectorReference:
    lap_length: float
    sector_length: float
    reference_lap_steps: int
    reference_lap_time: float
    source: str
    sectors: tuple[SectorTarget, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SectorReference":
        return cls(
            lap_length=float(payload.get("lap_length", 3608.45)),
            sector_length=float(payload.get("sector_length", 50.0)),
            reference_lap_steps=int(payload["reference_lap_steps"]),
            reference_lap_time=float(payload.get("reference_lap_time", 0.0)),
            source=str(payload.get("source", "")),
            sectors=tuple(SectorTarget.from_mapping(item) for item in payload.get("sectors", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lap_length": self.lap_length,
            "sector_length": self.sector_length,
            "reference_lap_steps": self.reference_lap_steps,
            "reference_lap_time": self.reference_lap_time,
            "source": self.source,
            "sectors": [sector.to_dict() for sector in self.sectors],
        }

    @property
    def sector_count(self) -> int:
        return len(self.sectors)

    def sector_index(self, distance: float) -> int:
        distance = max(0.0, min(distance, self.lap_length - 1e-6))
        return min(int(distance // self.sector_length), self.sector_count - 1)

    def sector_for_distance(self, distance: float) -> SectorTarget:
        return self.sectors[self.sector_index(distance)]


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_telemetry_rows(path: str | Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "step": _float_or_zero(row.get("step")),
                    "dist_raced": _float_or_zero(row.get("dist_raced")),
                    "dist_from_start": _float_or_zero(row.get("dist_from_start")),
                    "speed_x": _float_or_zero(row.get("speed_x")),
                }
            )
    if not rows:
        raise ValueError(f"telemetry file has no rows: {path}")
    return rows


def build_reference_from_telemetry(
    telemetry_path: str | Path,
    sector_length: float = 50.0,
    lap_length: float = 3608.45,
    reference_lap_time: float = 0.0,
) -> SectorReference:
    rows = _read_telemetry_rows(telemetry_path)
    sector_count = int(math.ceil(lap_length / sector_length))
    boundaries = [min((index + 1) * sector_length, lap_length) for index in range(sector_count)]

    sectors: list[SectorTarget] = []
    previous_step = int(rows[0]["step"])
    previous_boundary = 0.0
    row_index = 0

    for sector_index, boundary in enumerate(boundaries):
        speeds: list[float] = []
        crossed_step = int(rows[-1]["step"])
        while row_index < len(rows):
            row = rows[row_index]
            distance = row["dist_raced"]
            if previous_boundary <= distance <= boundary:
                speeds.append(row["speed_x"])
            if distance >= boundary:
                crossed_step = int(row["step"])
                row_index += 1
                break
            row_index += 1

        reference_steps = max(1, crossed_step - previous_step)
        reference_speed = sum(speeds) / len(speeds) if speeds else 0.0
        sectors.append(
            SectorTarget(
                index=sector_index,
                start=previous_boundary,
                end=boundary,
                reference_steps=reference_steps,
                reference_speed=reference_speed,
            )
        )
        previous_step = crossed_step
        previous_boundary = boundary

    reference_lap_steps = max(1, int(rows[-1]["step"] - rows[0]["step"]))
    return SectorReference(
        lap_length=lap_length,
        sector_length=sector_length,
        reference_lap_steps=reference_lap_steps,
        reference_lap_time=reference_lap_time,
        source=str(telemetry_path),
        sectors=tuple(sectors),
    )


def load_sector_reference(path: str | Path) -> SectorReference:
    with Path(path).open("r", encoding="utf-8") as fh:
        return SectorReference.from_mapping(json.load(fh))


def save_sector_reference(reference: SectorReference, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(reference.to_dict(), fh, indent=2, sort_keys=True)
