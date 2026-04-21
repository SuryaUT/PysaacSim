"""Drive the sim with the C controller template. Proves the C bridge works.

Requires gcc (MinGW-w64 on Windows) on PATH.

Run:
    python -m PySaacSim.examples.run_c_controller
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySaacSim.control.c_bridge import CController
from PySaacSim.env.robot_env import RobotEnv


def main():
    c_src = Path(__file__).with_name("controller_template.c")
    controller = CController(c_src)
    env = RobotEnv(controller=controller, max_episode_steps=500)

    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(None)
        total_reward += reward
    print(f"Done. t_ms={info['t_ms']:.0f} reward={total_reward:.1f} "
          f"collided={info['collided']}")
    env.close()


if __name__ == "__main__":
    main()
