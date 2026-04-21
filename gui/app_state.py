"""Shared app state. All pages observe and mutate this single source of truth.

Signals let the track builder update the sim page's canvas, a calibration
edit in the robot builder re-render sensor rays live, etc.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from ..control.base import AbstractController
from ..sim.calibration import SensorCalibration
from ..sim.geometry import Segment, Vec2
from ..sim.world import build_default_world


@dataclass
class RobotSpec:
    """One robot on the track: spawn pose + which controller drives it."""
    id: int
    x: float          # cm
    y: float          # cm
    theta: float      # rad
    controller_id: str = "manual-drive"   # key into AppState.controllers or "rl"/"manual-*"
    color: str = "#4fc3f7"


@dataclass
class RobotDims:
    """Geometry/physics constants the user can tweak in the robot builder.
    Defaults come from PySaacSim.sim.constants but are copied here so the GUI can
    mutate them per-session without editing the module."""
    chassis_length_cm: float
    chassis_width_cm: float
    chassis_height_cm: float
    wheelbase_cm: float
    rear_track_cm: float
    front_wheel_diam_cm: float
    rear_wheel_diam_cm: float
    max_speed_cms: float
    mass_kg: float

    @classmethod
    def default(cls) -> "RobotDims":
        from ..sim import constants as C
        return cls(
            chassis_length_cm=C.CHASSIS_LENGTH_CM,
            chassis_width_cm=C.CHASSIS_WIDTH_CM,
            chassis_height_cm=C.CHASSIS_HEIGHT_CM,
            wheelbase_cm=C.WHEELBASE_CM,
            rear_track_cm=C.REAR_TRACK_CM,
            front_wheel_diam_cm=C.FRONT_WHEEL_DIAM_CM,
            rear_wheel_diam_cm=C.REAR_WHEEL_DIAM_CM,
            max_speed_cms=C.MAX_SPEED_CMS,
            mass_kg=C.MASS_KG,
        )

    def copy(self) -> "RobotDims":
        return copy.deepcopy(self)


class AppState(QObject):
    """Single shared model. Pages connect to the signals they care about."""

    world_changed = pyqtSignal()          # walls mutated
    robots_changed = pyqtSignal()         # list of robots or a single spec mutated
    calibration_changed = pyqtSignal()    # sensor calibration mutated
    dims_changed = pyqtSignal()           # robot dims mutated
    controllers_changed = pyqtSignal()    # set of registered controllers changed
    training_reward = pyqtSignal(int, float)   # (iteration, mean_episode_reward)
    training_weights = pyqtSignal(object, object)  # (W ndarray, b ndarray)
    training_state = pyqtSignal(str)      # "idle" | "running" | "paused" | "stopped"

    def __init__(self) -> None:
        super().__init__()
        self._world = build_default_world()
        self.calibration: SensorCalibration = SensorCalibration.default()
        self.dims: RobotDims = RobotDims.default()
        self.robots: list[RobotSpec] = []
        self.controllers: dict[str, AbstractController] = {}
        self._next_robot_id = 0

        # Seed with the default spawn so the sim page isn't empty on launch.
        from ..sim.world import DEFAULT_SPAWN
        self.add_robot(DEFAULT_SPAWN["x"], DEFAULT_SPAWN["y"], DEFAULT_SPAWN["theta"])

    # ---- World / walls ----------------------------------------------------

    @property
    def walls(self) -> list[Segment]:
        return self._world["walls"]

    @property
    def world_bounds(self) -> dict:
        return self._world["bounds"]

    def set_walls(self, walls: list[Segment]) -> None:
        self._world["walls"] = list(walls)
        # Recompute bounds so the canvas auto-fits to user tracks.
        if walls:
            xs = [p.x for s in walls for p in (s.a, s.b)]
            ys = [p.y for s in walls for p in (s.a, s.b)]
            pad = 20.0
            self._world["bounds"] = {
                "min_x": min(xs) - pad, "max_x": max(xs) + pad,
                "min_y": min(ys) - pad, "max_y": max(ys) + pad,
            }
        self.world_changed.emit()

    def reset_world(self) -> None:
        self._world = build_default_world()
        self.world_changed.emit()

    # ---- Robots -----------------------------------------------------------

    def add_robot(self, x: float, y: float, theta: float = 0.0,
                  controller_id: str = "manual-drive") -> RobotSpec:
        spec = RobotSpec(id=self._next_robot_id, x=x, y=y, theta=theta,
                         controller_id=controller_id)
        self._next_robot_id += 1
        self.robots.append(spec)
        self.robots_changed.emit()
        return spec

    def remove_robot(self, robot_id: int) -> None:
        self.robots = [r for r in self.robots if r.id != robot_id]
        self.robots_changed.emit()

    def clear_robots(self) -> None:
        self.robots = []
        self.robots_changed.emit()

    def update_robot(self, robot_id: int, **changes) -> None:
        for r in self.robots:
            if r.id == robot_id:
                for k, v in changes.items():
                    setattr(r, k, v)
                self.robots_changed.emit()
                return

    # ---- Controllers ------------------------------------------------------

    def register_controller(self, name: str, controller: AbstractController) -> None:
        existing = self.controllers.pop(name, None)
        if existing is not None:
            try:
                existing.close()
            except Exception:
                pass
        self.controllers[name] = controller
        self.controllers_changed.emit()

    def unregister_controller(self, name: str) -> None:
        existing = self.controllers.pop(name, None)
        if existing is not None:
            try:
                existing.close()
            except Exception:
                pass
            # Any robots pointing at this controller fall back to "rl".
            for r in self.robots:
                if r.controller_id == name:
                    r.controller_id = "rl"
            self.controllers_changed.emit()
            self.robots_changed.emit()

    # ---- Calibration / dims ----------------------------------------------

    def set_calibration(self, cal: SensorCalibration) -> None:
        self.calibration = cal.copy()
        self.calibration_changed.emit()

    def set_dims(self, dims: RobotDims) -> None:
        self.dims = dims.copy()
        self.dims_changed.emit()
