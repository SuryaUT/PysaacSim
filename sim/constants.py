"""Physical constants — chassis geometry, dynamics, motor/servo limits.
Sensor calibration lives in calibration.py (YAML-driven). Constants here
describe the unchanging physics of the Lab 7 robot and are only edited if the
hardware itself is modified."""
from __future__ import annotations

import math


# -- Geometry (cm) ----------------------------------------------------------
CHASSIS_WIDTH_CM = 19.05
CHASSIS_LENGTH_CM = 30.48
CHASSIS_HEIGHT_CM = 13.97

FRONT_WHEEL_DIAM_CM = 7.62
REAR_WHEEL_DIAM_CM = 6.985

FRONT_AXLE_FROM_FRONT_CM = 10.16
REAR_AXLE_FROM_REAR_CM = 5.08
WHEELBASE_CM = CHASSIS_LENGTH_CM - FRONT_AXLE_FROM_FRONT_CM - REAR_AXLE_FROM_REAR_CM
REAR_TRACK_CM = CHASSIS_WIDTH_CM - 2.0


# -- Steering (MG996R) ------------------------------------------------------
SERVO_TIMER_HZ = 2_000_000
SERVO_MIN_COUNT = 2000
SERVO_CENTER_COUNT = 3000
SERVO_MAX_COUNT = 4000
STEER_LIMIT_RAD = math.radians(30)
SERVO_RAD_PER_SEC = math.radians(60) / 0.17


def servo_count_to_steer_angle(count: int) -> float:
    """Servo PWM count (2000..4000) -> wheel angle (rad). Right turn is negative
    (CCW-positive yaw convention in sim, matching firmware's sign convention)."""
    c = max(SERVO_MIN_COUNT, min(SERVO_MAX_COUNT, count))
    t = (c - SERVO_CENTER_COUNT) / (SERVO_MAX_COUNT - SERVO_CENTER_COUNT)
    return -t * STEER_LIMIT_RAD


# -- Motors -----------------------------------------------------------------
MOTOR_PWM_MAX_COUNT = 10000
MOTOR_PWM_HZ = 200
MOTOR_MAX_FORCE_N = 5.0
MOTOR_LAG_TAU_S = 0.100

# Top speed governor: measured 0.47 m/s at 9000/10000 duty on the real robot.
MAX_SPEED_CMS = 47


def duty_count_to_pwm01(count: int) -> float:
    if count <= 0:
        return 0.0
    if count >= MOTOR_PWM_MAX_COUNT:
        return 1.0
    return count / MOTOR_PWM_MAX_COUNT


# -- Dynamics ---------------------------------------------------------------
MASS_KG = 1.449
_w_m = CHASSIS_WIDTH_CM / 100.0
_l_m = CHASSIS_LENGTH_CM / 100.0
INERTIA_KG_M2 = MASS_KG * (_w_m * _w_m + _l_m * _l_m) / 12.0

MU_STATIC = 0.8
MU_KINETIC = 0.6
ROLLING_RESIST = 0.02
LINEAR_DRAG = 0.1
G = 9.81


# -- Timing -----------------------------------------------------------------
PHYSICS_DT_S = 0.001
RENDER_RATE_HZ = 60
IR_SAMPLE_RATE_HZ = 1000
