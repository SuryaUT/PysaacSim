"""Shared multi-robot execution engine."""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

from .constants import (
    MAX_SPEED_CMS, MOTOR_PWM_MAX_COUNT, PHYSICS_DT_S, SERVO_CENTER_COUNT,
    SERVO_MAX_COUNT, SERVO_MIN_COUNT, STEER_LIMIT_RAD,
)
from .geometry import Segment, chassis_segments
from .physics import RobotState, apply_command, initial_robot, step_physics
from .sensors import sample_sensors
from .state import RobotDims, RobotSpec
from .calibration import SensorCalibration
from ..control.base import AbstractController


# Same logic as gui/pages/simulation.py
CTRL_PERIOD_MS = 80.0
THROTTLE_MIN = 0.1  # Matches RobotEnv


def obs_from_sensors(sensors: dict, state: RobotState,
                     lidar_max: float, ir_max: float) -> np.ndarray:
    lc = sensors["lidar"]["center"]["distance_cm"] / lidar_max
    ll = sensors["lidar"]["left"]["distance_cm"] / lidar_max
    lr = sensors["lidar"]["right"]["distance_cm"] / lidar_max
    il = sensors["ir"]["left"]["distance_cm"] / ir_max
    ir = sensors["ir"]["right"]["distance_cm"] / ir_max
    v = state.v / MAX_SPEED_CMS
    om = max(-1.0, min(1.0, state.omega / 5.0))
    st = state.steer_angle / STEER_LIMIT_RAD
    return np.array([lc, ll, lr, il, ir, v, om, st], dtype=np.float32)


def action_to_command(servo_norm: float, tL_norm: float, tR_norm: float):
    servo_norm = max(-1.0, min(1.0, float(servo_norm)))
    tL_norm    = max(-1.0, min(1.0, float(tL_norm)))
    tR_norm    = max(-1.0, min(1.0, float(tR_norm)))
    if servo_norm >= 0:
        servo = int(SERVO_CENTER_COUNT + servo_norm * (SERVO_MAX_COUNT - SERVO_CENTER_COUNT))
    else:
        servo = int(SERVO_CENTER_COUNT + servo_norm * (SERVO_CENTER_COUNT - SERVO_MIN_COUNT))
    throttle_l = THROTTLE_MIN + (tL_norm + 1.0) * 0.5 * (1.0 - THROTTLE_MIN)
    throttle_r = THROTTLE_MIN + (tR_norm + 1.0) * 0.5 * (1.0 - THROTTLE_MIN)
    duty_l = int(throttle_l * MOTOR_PWM_MAX_COUNT)
    duty_r = int(throttle_r * MOTOR_PWM_MAX_COUNT)
    return duty_l, duty_r, servo


class SimEngine:
    """Manages the physics stepping and controller execution for N robots."""

    def __init__(self):
        self.runtimes: dict[int, RobotState] = {}
        self.steps_since_ctrl: dict[int, float] = {}
        
        # Playback configuration
        self.time_scale: float = 1.0
        self.cars_interact: bool = True
        self.auto_respawn: bool = True
        
        # External dependencies
        self.controllers: dict[str, AbstractController] = {}
        self.rl_policy: Optional[Callable[[np.ndarray], tuple[float, float, float]]] = None

        # Pre-calculated sensor buffers (for telemetry/WebSockets)
        self.last_sensors: dict[int, dict] = {}

    def reset_runtimes(self, robots: list[RobotSpec]) -> None:
        self.runtimes.clear()
        self.steps_since_ctrl.clear()
        self.ensure_runtimes(robots)

    def force_robot_pose(self, rid: int, x: float, y: float, theta: float) -> None:
        if rid in self.runtimes:
            self.runtimes[rid] = initial_robot(x, y, theta)
            self.steps_since_ctrl[rid] = 0.0

    def ensure_runtimes(self, robots: list[RobotSpec]) -> None:
        for r in robots:
            if r.id not in self.runtimes:
                self.runtimes[r.id] = initial_robot(r.x, r.y, r.theta)
                self.steps_since_ctrl[r.id] = 0.0
        live_ids = {r.id for r in robots}
        for rid in list(self.runtimes.keys()):
            if rid not in live_ids:
                self.runtimes.pop(rid, None)
                self.steps_since_ctrl.pop(rid, None)

    def _other_car_segments(self, me_id: int, robots: list[RobotSpec], dims: RobotDims) -> list[Segment]:
        out = []
        for other in robots:
            if other.id == me_id:
                continue
            rt = self.runtimes.get(other.id)
            if rt is None:
                continue
            pose = (rt.pose.x, rt.pose.y, rt.pose.theta)
            out.extend(chassis_segments(pose, dims.chassis_length_cm,
                                        dims.chassis_width_cm))
        return out

    def _compute_command(self, spec: RobotSpec, state: RobotState, sensors: dict, cal: SensorCalibration) -> tuple[int, int, int]:
        cid = spec.controller_id
        if cid == "manual-stop":
            return 0, 0, SERVO_CENTER_COUNT
        if cid == "manual-drive":
            return 9000, 9000, SERVO_CENTER_COUNT
        if cid == "rl":
            if self.rl_policy is None:
                return 0, 0, SERVO_CENTER_COUNT
            obs = obs_from_sensors(sensors, state, cal.lidar.max_cm, cal.ir.max_cm)
            action = self.rl_policy(obs)
            return action_to_command(float(action[0]), float(action[1]), float(action[2]))
        
        ctrl = self.controllers.get(cid)
        if ctrl is None:
            return 0, 0, SERVO_CENTER_COUNT
        cmd = ctrl.tick(sensors, 0.0)
        return cmd.duty_l, cmd.duty_r, cmd.servo

    def tick(self, dt_s: float, robots: list[RobotSpec], walls: list[Segment], 
             dims: RobotDims, cal: SensorCalibration) -> None:
        """Advance physics by exactly dt_s for all robots."""
        self.ensure_runtimes(robots)

        for r in robots:
            state = self.runtimes[r.id]
            if self.cars_interact:
                other_walls = self._other_car_segments(r.id, robots, dims)
                effective_walls = walls + other_walls
            else:
                effective_walls = walls
                
            self.steps_since_ctrl[r.id] += dt_s * 1000.0
            if self.steps_since_ctrl[r.id] >= CTRL_PERIOD_MS:
                self.steps_since_ctrl[r.id] = 0.0
                sensors = sample_sensors(effective_walls, state.pose, cal)
                duty_l, duty_r, servo = self._compute_command(r, state, sensors, cal)
                apply_command(state, duty_l, duty_r, servo)
                
            step_physics(state, effective_walls, dims.chassis_length_cm,
                         dims.chassis_width_cm, dt_s)
                         
            if state.collided and r.controller_id == "rl" and self.auto_respawn:
                self.runtimes[r.id] = initial_robot(r.x, r.y, r.theta)
                self.steps_since_ctrl[r.id] = 0.0
                state = self.runtimes[r.id]
                
    def tick_hz(self, hz: int, robots: list[RobotSpec], walls: list[Segment], 
                dims: RobotDims, cal: SensorCalibration) -> None:
        """Advance physics according to the GUI/Server loop tick rate and time_scale."""
        dt = PHYSICS_DT_S
        sim_seconds = self.time_scale / hz
        n_phys = max(1, int(round(sim_seconds / dt)))
        
        for _ in range(n_phys):
            self.tick(dt, robots, walls, dims, cal)
            
        # After physics, cache the latest sensor readings for rendering/telemetry
        for r in robots:
            state = self.runtimes[r.id]
            eff_walls = walls
            if self.cars_interact:
                eff_walls = walls + self._other_car_segments(r.id, robots, dims)
            self.last_sensors[r.id] = sample_sensors(eff_walls, state.pose, cal)
