"""Estimate static IMU bias from a motors-on or motors-off stationary capture.

The output is raw int16 LSB offsets per axis, matching what the firmware
would need to subtract to zero the sensor at rest. For `IMU_steady_state.csv`
the motors are on (throttle=9999 throughout), so the signal contains motor
vibration; the mean still gives a good bias estimate, but accel_y will
include any pitch-contribution if the chassis isn't perfectly level (the
plan flags accel_y ≈ -1100 LSB as likely partly-pitch).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .log_io import Log
from .windows import Window, find_quiescent


@dataclass
class IMUBiasEstimate:
    # x, y, z for each; we only log z for gyro (the others aren't in the log
    # format) — so gyro_x / gyro_y are always 0 from this estimate.
    gyro_bias: tuple[int, int, int]     # (x, y, z)
    accel_bias: tuple[int, int, int]    # (x, y, z)
    gyro_std_lsb: float                 # per-sample std of gyro_z
    accel_x_std_lsb: float
    accel_y_std_lsb: float
    n_samples: int
    notes: str


def estimate_bias(log: Log, window: Window | None = None) -> IMUBiasEstimate:
    if window is None:
        wins = find_quiescent(log)
        if not wins:
            raise ValueError(
                "no quiescent window found in log; pass an explicit Window"
            )
        # Take the longest one.
        window = max(wins, key=len)
    i, j = window.start, window.stop
    gz = log.gyro_z_lsb[i:j]
    ax = log.accel_x_lsb[i:j]
    ay = log.accel_y_lsb[i:j]
    gyro_bias_z = int(round(float(np.mean(gz))))
    accel_bias_x = int(round(float(np.mean(ax))))
    accel_bias_y = int(round(float(np.mean(ay))))
    # Caveat for the motors-on capture (accel_y ~ -1100 LSB suggests either
    # ~4° nose-up pitch or a real bias): flag when |accel_bias_y| is much
    # larger than the x/z biases, since that's the tell.
    notes = ""
    if abs(accel_bias_y) > 500 and abs(accel_bias_x) < 300:
        notes = (
            "WARNING: accel_y bias is much larger than accel_x — likely includes"
            " a pitch contribution, not a pure bias. Re-capture motors-off flat"
            " before trusting this value."
        )
    return IMUBiasEstimate(
        gyro_bias=(0, 0, gyro_bias_z),
        accel_bias=(accel_bias_x, accel_bias_y, 0),
        gyro_std_lsb=float(np.std(gz, ddof=1)) if gz.size > 1 else 0.0,
        accel_x_std_lsb=float(np.std(ax, ddof=1)) if ax.size > 1 else 0.0,
        accel_y_std_lsb=float(np.std(ay, ddof=1)) if ay.size > 1 else 0.0,
        n_samples=int(j - i),
        notes=notes,
    )
