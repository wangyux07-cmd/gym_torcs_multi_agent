from __future__ import annotations

import json
from pathlib import Path

from .controller import ControllerConfig


def load_controller_config(path: str | Path | None) -> ControllerConfig:
    if path is None:
        return ControllerConfig()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return ControllerConfig.from_mapping(payload)


def save_default_config(path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(ControllerConfig().to_dict(), fh, indent=2, sort_keys=True)

