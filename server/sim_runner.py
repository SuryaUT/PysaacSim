"""Headless simulation loop — powers the browser dashboard and multi-agent execution.

Design notes:
  * Zero PyQt dependency. Reuses ``sim.engine`` and ``sim.physics`` directly.
  * One singleton per server. The active track and any loaded policy are
    swappable at runtime.
  * Broadcasts pose + sensor snapshots for ALL robots at ~50 Hz.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
from typing import Any, Optional

from ..control.base import AbstractController, MotorCommand
from ..sim.calibration import SensorCalibration
from ..sim.constants import CHASSIS_LENGTH_CM, CHASSIS_WIDTH_CM, PHYSICS_DT_S
from ..sim.geometry import Segment, Vec2
from ..sim.physics import RobotState, initial_robot
from ..sim.sensors import sample_sensors
from ..sim.world import DEFAULT_SPAWN, build_default_world
from ..sim.state import RobotSpec, RobotDims
from ..sim.engine import SimEngine


logger = logging.getLogger(__name__)


def _segments_from_dict(walls: Any) -> list[Segment]:
    """Accept either a list of ``Segment`` (already loaded from sim.world) or
    JSON-decoded ``[[[ax,ay],[bx,by]], ...]``."""
    out: list[Segment] = []
    for w in walls:
        if isinstance(w, Segment):
            out.append(w)
            continue
        a, b = w[0], w[1]
        out.append(Segment(Vec2(float(a[0]), float(a[1])),
                           Vec2(float(b[0]), float(b[1]))))
    return out


class WebManualController(AbstractController):
    """Maps Web UI WASD keys directly to motor commands."""
    def __init__(self):
        self.cmd = MotorCommand()
        
    def tick(self, sensors: dict, t_ms: float) -> MotorCommand:
        return self.cmd
        
    def close(self):
        pass


class SimRunner:
    """Async sim loop. ``run()`` is the body; call ``set_command`` /
    ``load_track`` / robot API from request handlers to mutate state safely."""

    def __init__(self, ws_broadcaster, *, calibration: Optional[SensorCalibration] = None) -> None:
        self._broadcaster = ws_broadcaster
        self._cal = calibration or SensorCalibration.default()
        self._dims = RobotDims.default()

        world = build_default_world()
        self._walls: list[Segment] = list(world["walls"])
        self._spawn: dict[str, float] = dict(DEFAULT_SPAWN)

        self.engine = SimEngine()
        self.robots: list[RobotSpec] = []
        self.controllers: dict[str, AbstractController] = {}
        self._next_robot_id = 0

        self._running: bool = False
        self._task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()
        
        # Setup the default web keyboard controller
        self.web_manual = WebManualController()
        self.controllers["web-manual"] = self.web_manual
        self.add_robot(self._spawn["x"], self._spawn["y"], 
                       float(self._spawn.get("theta", 0.0)), controller_id="web-manual")

    # ---- Robot API --------------------------------------------------------

    def add_robot(self, x: float, y: float, theta: float = 0.0,
                  controller_id: str = "web-manual") -> RobotSpec:
        spec = RobotSpec(id=self._next_robot_id, x=x, y=y, theta=theta,
                         controller_id=controller_id)
        self._next_robot_id += 1
        self.robots.append(spec)
        return spec

    def remove_robot(self, robot_id: int) -> None:
        self.robots = [r for r in self.robots if r.id != robot_id]

    def clear_robots(self) -> None:
        self.robots = []

    def update_robot(self, robot_id: int, **changes) -> None:
        for r in self.robots:
            if r.id == robot_id:
                for k, v in changes.items():
                    setattr(r, k, v)
                return

    # ---- public API -------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def set_command(self, throttle: float, steer: float) -> None:
        """Manual drive override from the Web UI."""
        throttle = max(-1.0, min(1.0, float(throttle)))
        steer = max(-1.0, min(1.0, float(steer)))
        from ..sim.constants import (
            MOTOR_PWM_MAX_COUNT, SERVO_CENTER_COUNT, SERVO_MAX_COUNT, SERVO_MIN_COUNT,
        )
        if steer >= 0:
            servo = int(SERVO_CENTER_COUNT + steer * (SERVO_MAX_COUNT - SERVO_CENTER_COUNT))
        else:
            servo = int(SERVO_CENTER_COUNT + steer * (SERVO_CENTER_COUNT - SERVO_MIN_COUNT))
        duty = int(abs(throttle) * MOTOR_PWM_MAX_COUNT)
        dir_sign = 1 if throttle >= 0 else -1
        async with self._lock:
            self.web_manual.cmd = MotorCommand(duty_l=duty, duty_r=duty, servo=servo,
                                               dir_l=dir_sign, dir_r=dir_sign)

    async def load_track(self, track: dict[str, Any]) -> None:
        walls = _segments_from_dict(track["walls"])
        spawn = dict(track.get("spawn") or DEFAULT_SPAWN)
        async with self._lock:
            self._walls = walls
            self._spawn = spawn
            # Snap all existing robots back to the new spawn
            for r in self.robots:
                r.x = spawn["x"]
                r.y = spawn["y"]
                r.theta = spawn.get("theta", 0.0)
                self.engine.force_robot_pose(r.id, r.x, r.y, r.theta)

    async def reset(self) -> None:
        async with self._lock:
            self.engine.reset_runtimes(self.robots)
            self.web_manual.cmd = MotorCommand()

    async def load_policy(self, policy_path: str) -> None:
        """Load an SB3 PPO checkpoint and switch RL robots to autonomous mode."""
        from stable_baselines3 import PPO
        from pathlib import Path

        p = Path(policy_path)
        load_target = str(p.with_suffix("") if p.suffix == ".zip" else p)
        model = await asyncio.get_event_loop().run_in_executor(
            None, lambda: PPO.load(load_target, device="cpu")
        )
        async with self._lock:
            def ppo_policy(obs):
                action, _ = model.predict(obs, deterministic=True)
                return action
            self.engine.rl_policy = ppo_policy

    async def load_c_controller(self, name: str, code: str) -> str:
        """Compile and load a C controller from source string."""
        from ..control.c_bridge import CController
        import tempfile
        from pathlib import Path
        
        tmp = Path(tempfile.mkdtemp(prefix="pysim_cctrl_")) / f"{name}.c"
        tmp.write_text(code)
        
        async with self._lock:
            try:
                ctrl = CController(tmp)
                ctrl.init()
                if name in self.controllers:
                    try: self.controllers[name].close()
                    except: pass
                self.controllers[name] = ctrl
                return "ok"
            except Exception as e:
                return str(e)
                
    async def unload_controller(self, name: str) -> None:
        async with self._lock:
            if name in self.controllers:
                try: self.controllers[name].close()
                except: pass
                del self.controllers[name]
                for r in self.robots:
                    if r.controller_id == name:
                        r.controller_id = "rl"

    async def unload_policy(self) -> None:
        async with self._lock:
            self.engine.rl_policy = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "walls": [[list(w.a), list(w.b)] for w in self._walls],
            "spawn": self._spawn,
        }

    # ---- loop -------------------------------------------------------------

    async def _run(self) -> None:
        # 50 Hz control: 20 ms per tick
        ctrl_dt_s = 0.02
        try:
            while self._running:
                if self._broadcaster.client_count() > 0:
                    async with self._lock:
                        self.engine.controllers = self.controllers
                        # Perform exactly 1 frame of physics at 50Hz (scaled by time_scale)
                        self.engine.tick_hz(int(1.0 / ctrl_dt_s), self.robots, self._walls, self._dims, self._cal)
                        payload = self._snapshot_msg()
                    await self._broadcaster.broadcast(json.dumps(payload, separators=(",", ":")))
                await asyncio.sleep(ctrl_dt_s)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Sim runner crashed")

    def _snapshot_msg(self) -> dict[str, Any]:
        def _flat(reading: dict[str, Any]) -> dict[str, Any]:
            hit = reading.get("hit")
            return {
                "distance_cm": float(reading["distance_cm"]),
                "valid": bool(reading["valid"]),
                "origin": [reading["origin"].x, reading["origin"].y],
                "hit": [hit.x, hit.y] if hit is not None else None,
            }

        robots_data = []
        for r in self.robots:
            state = self.engine.runtimes.get(r.id)
            if not state: continue
            sensors = self.engine.last_sensors.get(r.id, {})
            lidar = sensors.get("lidar", {})
            ir = sensors.get("ir", {})
            
            robots_data.append({
                "id": r.id,
                "controller_id": r.controller_id,
                "color": r.color,
                "pose": {"x": state.pose.x, "y": state.pose.y, "theta": state.pose.theta},
                "v": state.v,
                "omega": state.omega,
                "steer": state.steer_angle,
                "collided": bool(state.collided),
                "lidar": {k: _flat(v) for k, v in lidar.items()} if lidar else {},
                "ir": {k: _flat(v) for k, v in ir.items()} if ir else {},
            })

        return {
            "kind": "sim",
            "robots": robots_data,
        }
