"""Sensor calibration surface — all tunable knobs that describe real-hardware
behavior, loaded from YAML so no code changes are needed when the physical
robot's calibration is updated.

One knob per physical quantity:
  IR:
    curve_k, curve_c     analog front-end response v(d_cm) = k / (d_cm + c)
    peak_cm              non-monotonic regime boundary
    min_cm, max_cm       reported-valid range
    *_noise_std          gaussian jitter
    adc_*                ADC converter params
  Lidar:
    max_cm, noise, scale/bias, rate, fov
  Placements:
    per-sensor (x, y, theta) in body frame
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Union

import yaml


_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "config" / "calibration.yaml"


@dataclass
class IRCalibration:
    # v(d_cm) = curve_k / (d_cm + curve_c) for d_cm >= peak_cm.
    # Defaults reproduce the Lab 7 analog front-end doubling: IRDistance_Convert
    # returns 2 x true_mm, and firmware /2 in DAS recovers true mm. Change
    # curve_k (e.g., to 25.58) if the hardware response changes.
    curve_k: float = 12.79
    curve_c: float = 0.585
    peak_cm: float = 4.0
    peak_floor_v: float = 0.4
    v_clamp: float = 3.1
    min_cm: float = 4.0
    max_cm: float = 30.0
    distance_noise_std_cm: float = 0.4
    voltage_noise_std: float = 0.02
    adc_max_count: int = 4095
    adc_vref: float = 3.3


@dataclass
class LidarCalibration:
    max_cm: float = 800.0
    noise_std_cm: float = 2.0
    fov_rad: float = math.radians(2)
    rate_hz: float = 100.0
    # reported_cm = raw_cm * distance_scale + distance_bias_cm
    distance_scale: float = 1.0
    distance_bias_cm: float = 0.0


@dataclass
class SensorPlacement:
    id: str
    x: float
    y: float
    theta: float


@dataclass
class SensorCalibration:
    ir: IRCalibration
    lidar: LidarCalibration
    lidar_placements: list[SensorPlacement]
    ir_placements: list[SensorPlacement]

    @classmethod
    def default(cls) -> "SensorCalibration":
        return cls.from_yaml(_DEFAULT_YAML)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SensorCalibration":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "SensorCalibration":
        ir = IRCalibration(**data["ir"])
        lidar = LidarCalibration(**data["lidar"])
        lidars = [SensorPlacement(**p) for p in data["lidar_placements"]]
        irs = [SensorPlacement(**p) for p in data["ir_placements"]]
        if len(lidars) != 3:
            raise ValueError(f"expected 3 lidar placements, got {len(lidars)}")
        if len(irs) != 2:
            raise ValueError(f"expected 2 IR placements, got {len(irs)}")
        return cls(ir=ir, lidar=lidar, lidar_placements=lidars, ir_placements=irs)

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    def copy(self) -> "SensorCalibration":
        return copy.deepcopy(self)


def ir_distance_to_volts(d_cm: float, cal: IRCalibration) -> float:
    """Voltage produced by the IR analog front-end for a true distance d_cm.

    Below peak_cm the sensor is non-monotonic (very-close object reads like a
    ~25 cm object); model that regime as a linear rise from peak_floor_v at
    d=0 up to the rational-curve value at peak_cm.
    """
    peak_v = cal.curve_k / (cal.peak_cm + cal.curve_c)
    clamped_peak = min(cal.v_clamp, peak_v)
    if d_cm < cal.peak_cm:
        t = d_cm / cal.peak_cm
        return cal.peak_floor_v + (clamped_peak - cal.peak_floor_v) * t
    v = cal.curve_k / (d_cm + cal.curve_c)
    return max(0.0, min(cal.v_clamp, v))
