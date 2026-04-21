"""PPO training example. Trains a wall-following policy in parallel on 8
robot envs. Save tensorboard logs to ./tb_logs.

Run:
    python -m PySaacSim.examples.train_ppo
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from PySaacSim.env.robot_env import RobotEnv


def make_env():
    return RobotEnv()


def main():
    vec = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)
    model = PPO(
        "MlpPolicy", vec,
        verbose=1,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        tensorboard_log="./tb_logs",
    )
    model.learn(total_timesteps=1_000_000)
    model.save("wall_follower_ppo")


if __name__ == "__main__":
    main()
