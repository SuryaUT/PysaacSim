"""Linear policy helpers for SB3 PPO with `net_arch=[]`.

With net_arch=[], the policy becomes:
    action = W @ obs + b
(plus a stochastic term during training; deterministic at rollout time).

Feature layout matches PySaacSim.env.RobotEnv.observation_space:
    [lidar_c, lidar_l, lidar_r, ir_l, ir_r, v, omega, steer]   (8 features)
Action layout:
    [servo_norm, throttle_L_norm, throttle_R_norm] all ∈ [-1,1]  (3 actions)
Each throttle is remapped by the env to [THROTTLE_MIN, 1.0]. Two
throttles match the firmware's differential-drive CAN_SetMotors
(separate duty for left and right rear wheels).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


FEATURE_NAMES = ["lidar_c", "lidar_l", "lidar_r", "ir_l", "ir_r",
                 "v", "omega", "steer"]
ACTION_NAMES = ["servo", "throttle_L", "throttle_R"]


def extract_weights(model) -> tuple[np.ndarray, np.ndarray]:
    """Pull the linear policy (W, b) out of a trained SB3 PPO model.

    Returns W shape (action_dim, obs_dim) and b shape (action_dim,)."""
    action_net = model.policy.action_net
    W = action_net.weight.detach().cpu().numpy().copy()
    b = action_net.bias.detach().cpu().numpy().copy()
    return W, b


def policy_forward(W: np.ndarray, b: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Deterministic forward pass. obs shape (8,) -> action shape (3,).

    [servo_norm, tL_norm, tR_norm], all clipped to [-1, +1]. The env remaps
    each throttle → [THROTTLE_MIN, 1.0]."""
    a = W @ obs + b
    for i in range(a.shape[0]):
        a[i] = max(-1.0, min(1.0, a[i]))
    return a
