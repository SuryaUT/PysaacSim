"""Tiny matplotlib widget embedded in Qt — used for the reward curve."""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QSizePolicy


class RewardCurve(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(figsize=(4, 2), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._ax = fig.add_subplot(111)
        self._xs: list[int] = []
        self._ys: list[float] = []
        self._ax.set_xlabel("timesteps")
        self._ax.set_ylabel("mean ep reward")
        self._ax.grid(True, alpha=0.3)
        self._line, = self._ax.plot([], [], color="#2196f3")

    def add_point(self, x: int, y: float) -> None:
        self._xs.append(x)
        self._ys.append(y)
        self._line.set_data(self._xs, self._ys)
        self._ax.relim()
        self._ax.autoscale_view()
        self.draw_idle()

    def clear(self) -> None:
        self._xs = []
        self._ys = []
        self._line.set_data([], [])
        self.draw_idle()


class WeightHeatmap(FigureCanvasQTAgg):
    """Show the W matrix (action_dim × feature_dim) as a heatmap."""
    def __init__(self, feature_names: list[str], action_names: list[str], parent=None):
        fig = Figure(figsize=(4, 2), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self._feat = feature_names
        self._act = action_names
        self._ax = fig.add_subplot(111)
        self._im = None

    def update_weights(self, W, b=None) -> None:
        self._ax.clear()
        self._im = self._ax.imshow(W, aspect="auto", cmap="RdBu_r",
                                   vmin=-max(1e-6, float(abs(W).max())),
                                   vmax=+max(1e-6, float(abs(W).max())))
        self._ax.set_yticks(range(len(self._act)))
        self._ax.set_yticklabels(self._act)
        self._ax.set_xticks(range(len(self._feat)))
        self._ax.set_xticklabels(self._feat, rotation=35, ha="right", fontsize=8)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                self._ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                              fontsize=7, color="#111")
        self.draw_idle()
