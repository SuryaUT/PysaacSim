"""QThread that runs PPO training in the background and reports progress.

Uses a single in-process RobotEnv (not VecEnv) so training is slow but
observable. For production-speed training, use examples/train_ppo.py.

Emits:
  iteration(int, mean_reward float)       every PPO update
  weights(W ndarray, b ndarray)           every PPO update
  state_changed(str)                      "running" | "paused" | "stopped" | "done"
  log(str)                                user-facing status strings
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from ...env.robot_env import RobotEnv
from ...sim.calibration import SensorCalibration
from ...sim.geometry import Segment
from .linear_policy import extract_weights


class _CallbackBridge:
    """Adapter hooked into SB3 via a BaseCallback subclass created inside run()."""
    def __init__(self, worker: "TrainingWorker"):
        self.worker = worker


class TrainingWorker(QThread):
    iteration = pyqtSignal(int, float)
    weights = pyqtSignal(object, object)  # np.ndarray, np.ndarray
    state_changed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, walls: list[Segment], calibration: SensorCalibration,
                 total_timesteps: int = 200_000,
                 n_steps: int = 512, learning_rate: float = 3e-4,
                 device: str = "cpu",
                 n_envs: int = 1,
                 same_scene: bool = False,
                 save_path: Optional[str] = None,
                 save_every: int = 50_000,
                 resume: bool = False,
                 parent=None):
        super().__init__(parent)
        self._walls = list(walls)
        self._cal = calibration.copy()
        self._total = total_timesteps
        self._n_steps = n_steps
        self._lr = learning_rate
        self._device = device
        self._n_envs = max(1, int(n_envs))
        self._same_scene = bool(same_scene)
        self._save_path = Path(save_path) if save_path else None
        self._save_every = max(0, int(save_every))
        self._resume = bool(resume)
        self._stop_flag = False
        self._pause_flag = False
        self._W: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None

    # ---- controls ---------------------------------------------------------

    def request_stop(self) -> None:
        self._stop_flag = True

    def request_pause(self, pause: bool) -> None:
        self._pause_flag = pause
        self.state_changed.emit("paused" if pause else "running")

    def latest_weights(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self._W, self._b

    # ---- thread body ------------------------------------------------------

    def run(self) -> None:
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as e:
            self.log.emit(f"stable-baselines3 not installed ({e}). "
                          f"Run: pip install stable-baselines3")
            self.state_changed.emit("stopped")
            return
        except OSError as e:
            self.log.emit(
                f"Torch failed to load its DLLs: {e}\n"
                f"Fix: pip install --force-reinstall torch")
            self.state_changed.emit("stopped")
            return
        except Exception as e:
            self.log.emit(f"Training stack import error: {type(e).__name__}: {e}")
            self.state_changed.emit("stopped")
            return

        try:
            self._do_train(PPO, BaseCallback)
        except Exception as e:
            import traceback
            self.log.emit(f"Training crashed: {type(e).__name__}: {e}\n"
                          f"{traceback.format_exc()}")
            self.state_changed.emit("stopped")

    def _do_train(self, PPO, BaseCallback) -> None:
        cal = self._cal
        walls = self._walls

        # Monitor / VecMonitor wrap env so PPO's ep_info_buffer gets populated.
        # Without this, mean episode reward shows as 0 forever even if episodes
        # are completing (crashes + timeouts), because SB3 only records episode
        # returns that pass through the Monitor wrapper.
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

        if self._same_scene and self._n_envs > 1:
            # All N cars share one track, see each other as obstacles.
            from ...env.multi_robot_env import MultiRobotVecEnv
            self.log.emit(f"Building same-scene MultiRobotVecEnv with "
                          f"{self._n_envs} agents…")
            env = MultiRobotVecEnv(n_agents=self._n_envs,
                                   walls=walls, calibration=cal,
                                   max_episode_steps=1500)
            env = VecMonitor(env)
        else:
            self.log.emit(f"Building {self._n_envs}× RobotEnv…")

            def _make_env():
                e = RobotEnv(calibration=cal, max_episode_steps=1500)
                e._world["walls"] = walls
                e._walls = walls
                return Monitor(e)

            if self._n_envs == 1:
                env = _make_env()
            else:
                env = DummyVecEnv([(lambda: _make_env()) for _ in range(self._n_envs)])
        self.log.emit(f"Env ready. n_envs={self._n_envs} "
                      f"same_scene={self._same_scene} walls={len(walls)}")

        worker = self
        save_path = self._save_path
        save_every = self._save_every

        class ProgressCb(BaseCallback):
            def __init__(self):
                super().__init__()
                self._next_save_step = save_every if (save_path and save_every > 0) else None

            def _on_step(self) -> bool:
                while worker._pause_flag and not worker._stop_flag:
                    worker.msleep(50)
                if worker._stop_flag:
                    return False
                if self._next_save_step is not None and self.num_timesteps >= self._next_save_step:
                    try:
                        self.model.save(str(save_path))
                        worker.log.emit(f"[checkpoint] saved @ {self.num_timesteps:,} steps → {save_path}.zip")
                    except Exception as e:
                        worker.log.emit(f"[checkpoint] save failed: {type(e).__name__}: {e}")
                    self._next_save_step += save_every
                return True

            def _on_rollout_end(self) -> None:
                W, b = extract_weights(self.model)
                worker._W, worker._b = W, b
                # Mean episode reward from the logger's ep_info_buffer.
                mean_r = 0.0
                ep_buf = self.model.ep_info_buffer
                if ep_buf:
                    mean_r = float(np.mean([ep["r"] for ep in ep_buf]))
                worker.weights.emit(W, b)
                worker.iteration.emit(self.num_timesteps, mean_r)

        self.state_changed.emit("running")

        # --- Load existing model or construct fresh ------------------------
        model = None
        if self._resume and self._save_path is not None:
            zip_path = self._save_path.with_suffix(".zip")
            if zip_path.exists():
                try:
                    self.log.emit(f"Resuming from {zip_path} …")
                    model = PPO.load(str(self._save_path), env=env,
                                     device=self._device)
                    # Honor new LR from the UI even when resuming.
                    model.learning_rate = self._lr
                    model._setup_lr_schedule()
                except Exception as e:
                    self.log.emit(f"Resume failed ({type(e).__name__}: {e}); "
                                  f"starting fresh.")
                    model = None
            else:
                self.log.emit(f"No checkpoint at {zip_path}; starting fresh.")

        if model is None:
            self.log.emit(f"Constructing PPO (linear policy, total={self._total} steps)…")
            # For an 8-input / 2-output linear policy CPU is typically faster —
            # PCIe transfer per step dominates GPU compute. Configurable from GUI.
            model = PPO(
                "MlpPolicy", env,
                n_steps=self._n_steps,
                batch_size=min(64, self._n_steps),
                learning_rate=self._lr,
                policy_kwargs=dict(net_arch=[]),   # linear policy
                device=self._device,
                verbose=0,
            )

        if self._save_path is not None:
            # Make sure the parent dir exists before the first checkpoint fires.
            self._save_path.parent.mkdir(parents=True, exist_ok=True)

        self.log.emit(f"PPO ready. device={model.device}. starting learn()…")
        # Emit initial weights so the UI has something to show.
        W, b = extract_weights(model)
        self._W, self._b = W, b
        self.weights.emit(W, b)

        try:
            model.learn(total_timesteps=self._total, callback=ProgressCb(),
                        progress_bar=False, reset_num_timesteps=not self._resume)
        except Exception as e:
            self.log.emit(f"Training error: {e}")
        finally:
            # Final save regardless of clean finish vs. user stop.
            if self._save_path is not None:
                try:
                    model.save(str(self._save_path))
                    self.log.emit(f"[final] saved → {self._save_path}.zip")
                except Exception as e:
                    self.log.emit(f"[final] save failed: {type(e).__name__}: {e}")
            env.close()
            self.state_changed.emit("done" if not self._stop_flag else "stopped")
            self.log.emit("Training finished.")
