"""Pure-Python data structures for simulation state, shared by GUI and Server."""
from __future__ import annotations

import copy
from dataclasses import dataclass

@dataclass
class RobotSpec:
    """One robot on the track: spawn pose + which controller drives it."""
    id: int
    x: float          # cm
    y: float          # cm
    theta: float      # rad
    controller_id: str = "manual-drive"   # key into controllers dict or "rl"/"manual-*"
    color: str = "#4fc3f7"


@dataclass
class RobotDims:
    """Geometry/physics constants the user can tweak.
    Defaults come from PySaacSim.sim.constants."""
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
        from . import constants as C
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
