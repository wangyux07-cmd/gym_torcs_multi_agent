from __future__ import annotations

import argparse
from pathlib import Path

from race_ai.sector_reference import build_reference_from_telemetry, save_sector_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sector-time reference JSON from a telemetry CSV.")
    parser.add_argument("--telemetry", required=True, help="Path to telemetry_episode_*.csv from race_profile_driver.py")
    parser.add_argument("--out", default="configs/sector_reference_v2_12_50m.json")
    parser.add_argument("--sector-length", type=float, default=50.0)
    parser.add_argument("--lap-length", type=float, default=3608.45)
    parser.add_argument("--reference-lap-time", type=float, default=125.404)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = build_reference_from_telemetry(
        telemetry_path=args.telemetry,
        sector_length=args.sector_length,
        lap_length=args.lap_length,
        reference_lap_time=args.reference_lap_time,
    )
    save_sector_reference(reference, args.out)
    print(
        "wrote {path} sectors={sectors} reference_steps={steps} reference_lap_time={lap:.3f}".format(
            path=Path(args.out),
            sectors=reference.sector_count,
            steps=reference.reference_lap_steps,
            lap=reference.reference_lap_time,
        )
    )


if __name__ == "__main__":
    main()
