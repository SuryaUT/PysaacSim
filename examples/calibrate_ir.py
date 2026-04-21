"""Calibration workflow demo — edit sensor response without touching code.

Scenario: you measure the real IR sensor's voltage at three known distances,
discover the analog front-end gain has drifted, and want the sim to reflect
it. This script:

  1. Loads current calibration from config/calibration.yaml.
  2. Fits new curve_k/curve_c to your measurements.
  3. Writes the updated calibration back to a YAML file.
  4. Spins up a RobotEnv with the new calibration and prints a sensor reading
     to prove it took effect — no simulator code was modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from PySaacSim.sim.calibration import SensorCalibration
from PySaacSim.env.robot_env import RobotEnv


# Fake measurement data. Real workflow: aim the robot's IR at a wall, record
# voltage with a scope or ADC, move to the next distance, repeat.
MEASUREMENTS_CM_V = [
    (5.0, 2.28),
    (10.0, 1.21),
    (15.0, 0.82),
    (20.0, 0.62),
    (25.0, 0.50),
]


def fit_curve(meas: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit v = k / (d + c) via linear regression on 1/v vs d.
       1/v = (1/k)d + c/k → slope=1/k, intercept=c/k → c = intercept*k."""
    d = np.array([m[0] for m in meas])
    v = np.array([m[1] for m in meas])
    inv_v = 1.0 / v
    slope, intercept = np.polyfit(d, inv_v, 1)
    k = 1.0 / slope
    c = intercept * k
    return float(k), float(c)


def main():
    cal = SensorCalibration.default()
    print(f"Before: curve_k={cal.ir.curve_k:.3f}, curve_c={cal.ir.curve_c:.3f}")

    new_k, new_c = fit_curve(MEASUREMENTS_CM_V)
    print(f"Fit:    curve_k={new_k:.3f}, curve_c={new_c:.3f}")
    cal.ir.curve_k = new_k
    cal.ir.curve_c = new_c

    out = Path("measured_calibration.yaml")
    cal.to_yaml(out)
    print(f"Wrote   {out.resolve()}")

    # Prove the env uses the new calibration without any code changes.
    env = RobotEnv(calibration=cal)
    obs, _ = env.reset()
    print(f"\nReset obs (normalized IR readings idx 3,4): "
          f"ir_left={obs[3]:.3f}, ir_right={obs[4]:.3f}")


if __name__ == "__main__":
    main()
