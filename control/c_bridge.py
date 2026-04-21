"""Compile a user's C controller into a shared library and drive it via
ctypes.

Usage:
    ctrl = CController("my_controller.c")          # file path
    ctrl = CController(c_src_string)                # inline source
    ctrl.init()
    cmd = ctrl.tick(sensor_packet, t_ms)

The user's C file must implement `void robot_init(void)` and
`void robot_tick(void)`, including "hal.h" from PySaacSim/hal/. Python writes
input globals (IR + lidar distances in mm) before each tick and reads the
output globals written by CAN_SetMotors.

For multi-robot parallelism, spawn each env in its own subprocess
(gymnasium SubprocVecEnv) — each subprocess loads its own .dll, keeping
global C state isolated per-robot.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Union

from .base import AbstractController, MotorCommand


_HAL_DIR = Path(__file__).resolve().parent.parent / "hal"
_HAL_SOURCES = ["hal_impl.c", "filter.c", "irdistance.c"]


class CController(AbstractController):
    def __init__(self, c_src: Union[str, Path], gcc: str = "gcc", extra_cflags: list[str] | None = None):
        src_text, hint_name = _resolve_src(c_src)
        self._tmpdir = tempfile.mkdtemp(prefix="pysim_cctrl_")
        self._lib_path = self._build(src_text, hint_name, gcc, extra_cflags or [])
        self._lib = ctypes.CDLL(str(self._lib_path))
        self._bind_symbols()

    # -- Controller API -----------------------------------------------------

    def init(self) -> None:
        # Zero all sensor inputs and PD state before the first tick.
        self._set_input(0, 0, 0, 0, 0)
        self._t_ms.value = 0
        self._prev_error.value = 0
        self._prev_e_a.value = 0
        self._robot_init()

    def tick(self, sensors: dict, t_ms: float) -> MotorCommand:
        # Convert sim sensor dict -> firmware globals.
        # IR: ADC count -> IRDistance_Convert -> /2 (mirrors DAS pipeline).
        ir_r = sensors["ir"]["right"]
        ir_l = sensors["ir"]["left"]
        d_right = self._ir_distance_convert(ir_r["adc"]) // 2
        d_left  = self._ir_distance_convert(ir_l["adc"]) // 2

        # Lidar: cm -> mm. Use left/right tflunas for the wall-follower.
        lidar_right_mm = int(round(sensors["lidar"]["right"]["distance_cm"] * 10))
        lidar_left_mm  = int(round(sensors["lidar"]["left"]["distance_cm"] * 10))
        lidar_center_mm = int(round(sensors["lidar"]["center"]["distance_cm"] * 10))

        self._set_input(d_right, d_left, lidar_right_mm, lidar_left_mm, lidar_center_mm)
        self._t_ms.value = int(t_ms)
        self._robot_tick()

        return MotorCommand(
            duty_l=int(self._duty_l.value),
            duty_r=int(self._duty_r.value),
            servo=int(self._servo.value),
        )

    def close(self) -> None:
        # ctypes leaks the handle on Windows if not explicitly freed; best-effort.
        handle = getattr(self._lib, "_handle", None)
        if handle and sys.platform.startswith("win"):
            try:
                ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
            except Exception:
                pass
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # -- Internals ----------------------------------------------------------

    def _build(self, src_text: str, hint_name: str, gcc: str, extra_cflags: list[str]) -> Path:
        tmp = Path(self._tmpdir)
        user_c = tmp / "user_controller.c"
        user_c.write_text(src_text)

        # Copy HAL sources + header into the temp dir so relative #include "hal.h" works.
        for name in _HAL_SOURCES + ["hal.h", "hal_shim.h"]:
            shutil.copyfile(_HAL_DIR / name, tmp / name)

        lib_name = "controller.dll" if sys.platform.startswith("win") else "libcontroller.so"
        lib_path = tmp / lib_name

        cmd = [
            gcc, "-O2", "-shared", "-fPIC",
            "-o", str(lib_path),
            str(user_c),
            *[str(tmp / s) for s in _HAL_SOURCES],
            "-I", str(tmp),
            *extra_cflags,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"`{gcc}` not found on PATH. Install a C compiler:\n"
                "  Windows: winget install -e --id BrechtSanders.WinLibs.POSIX.MSVCRT.GCC.UCRT  (or MSYS2)\n"
                "  Linux:   apt install build-essential\n"
                "  macOS:   xcode-select --install"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"C compilation failed for '{hint_name}':\n{e.stderr}"
            ) from e
        return lib_path

    def _bind_symbols(self) -> None:
        lib = self._lib

        lib.robot_init.argtypes = []
        lib.robot_init.restype = None
        lib.robot_tick.argtypes = []
        lib.robot_tick.restype = None
        self._robot_init = lib.robot_init
        self._robot_tick = lib.robot_tick

        # IRDistance_Convert exported from irdistance.c (reused so Python mirrors
        # the firmware's DAS pipeline without duplicating the math).
        lib.IRDistance_Convert.argtypes = [ctypes.c_int32, ctypes.c_uint32]
        lib.IRDistance_Convert.restype = ctypes.c_int32
        self._ir_distance_convert = lambda adc: lib.IRDistance_Convert(int(adc), 0)

        # Input globals
        self._g_distance_raw   = ctypes.c_uint32.in_dll(lib, "g_DistanceRaw")
        self._g_l_distance_raw = ctypes.c_uint32.in_dll(lib, "g_L_DistanceRaw")
        self._g_distance2      = ctypes.c_uint32.in_dll(lib, "g_Distance2")
        self._g_l_distance2    = ctypes.c_uint32.in_dll(lib, "g_L_Distance2")
        self._g_distance_center = ctypes.c_uint32.in_dll(lib, "g_DistanceCenter")

        # Output globals
        self._duty_l = ctypes.c_uint16.in_dll(lib, "g_duty_l")
        self._duty_r = ctypes.c_uint16.in_dll(lib, "g_duty_r")
        self._servo  = ctypes.c_int16.in_dll(lib, "g_servo")

        # Time + PD state
        self._t_ms       = ctypes.c_uint32.in_dll(lib, "g_t_ms")
        self._prev_error = ctypes.c_int32.in_dll(lib, "prevError")
        self._prev_e_a   = ctypes.c_int32.in_dll(lib, "prevE_A")

    def _set_input(self, d_right_mm, d_left_mm, lidar_right_mm, lidar_left_mm, lidar_center_mm):
        self._g_distance_raw.value   = int(d_right_mm) & 0xFFFFFFFF
        self._g_l_distance_raw.value = int(d_left_mm) & 0xFFFFFFFF
        self._g_distance2.value      = int(lidar_right_mm) & 0xFFFFFFFF
        self._g_l_distance2.value    = int(lidar_left_mm) & 0xFFFFFFFF
        self._g_distance_center.value = int(lidar_center_mm) & 0xFFFFFFFF


def _resolve_src(c_src: Union[str, Path]) -> tuple[str, str]:
    """Accept either a .c file path or inline C source. Returns (text, display_name)."""
    if isinstance(c_src, Path):
        return c_src.read_text(), c_src.name
    if isinstance(c_src, str):
        p = Path(c_src)
        if len(c_src) < 260 and p.exists() and p.suffix == ".c":
            return p.read_text(), p.name
        return c_src, "<inline>"
    raise TypeError(f"c_src must be Path or str, got {type(c_src)}")
