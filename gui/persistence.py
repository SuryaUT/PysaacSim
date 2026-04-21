"""YAML save/load for custom tracks and robot dimensions."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Union

import yaml

from ..sim.geometry import Segment, Vec2
from .app_state import RobotDims


def save_track(walls: list[Segment], path: Union[str, Path]) -> None:
    data = {
        "walls": [
            {"ax": s.a.x, "ay": s.a.y, "bx": s.b.x, "by": s.b.y}
            for s in walls
        ],
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_track(path: Union[str, Path]) -> list[Segment]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return [
        Segment(Vec2(w["ax"], w["ay"]), Vec2(w["bx"], w["by"]))
        for w in data["walls"]
    ]


def save_dims(dims: RobotDims, path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(asdict(dims), f, sort_keys=False)


def load_dims(path: Union[str, Path]) -> RobotDims:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return RobotDims(**data)
