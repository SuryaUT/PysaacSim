"""Track builder: click-drag walls, endpoint handles, grid snap, YAML I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from ...sim.geometry import Segment, Vec2
from ..app_state import AppState
from ..canvas import TrackCanvas
from ..persistence import load_track, save_track


class _DrawableCanvas(TrackCanvas):
    """Canvas subclass that supports click-drag to create a new wall."""
    def __init__(self, state: AppState):
        super().__init__(state, mode="build")
        self._drawing: Optional[Vec2] = None
        self._preview = None

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(ev.pos())
            if self._snap_enabled:
                pos = self._snap_fn()(pos)
            item = self.itemAt(ev.pos())
            # If clicking empty space, start drawing a new wall.
            if item is None:
                self._drawing = Vec2(pos.x(), pos.y())
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._drawing is not None:
            pos = self.mapToScene(ev.pos())
            if self._snap_enabled:
                pos = self._snap_fn()(pos)
            # draw live preview line
            from PyQt6.QtWidgets import QGraphicsLineItem
            from PyQt6.QtGui import QPen, QColor
            if self._preview is None:
                self._preview = QGraphicsLineItem()
                thickness = float(self.state._world.get("wall_thickness_cm", 8.89))
                pen = QPen(QColor(233, 30, 99, 140), thickness)
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setCapStyle(Qt.PenCapStyle.FlatCap)
                self._preview.setPen(pen)
                self._scene.addItem(self._preview)
            self._preview.setLine(self._drawing.x, self._drawing.y, pos.x(), pos.y())
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and self._drawing is not None:
            pos = self.mapToScene(ev.pos())
            if self._snap_enabled:
                pos = self._snap_fn()(pos)
            start = self._drawing
            end = Vec2(pos.x(), pos.y())
            self._drawing = None
            if self._preview is not None:
                self._scene.removeItem(self._preview)
                self._preview = None
            # Require at least a 2 cm segment.
            if abs(end.x - start.x) + abs(end.y - start.y) > 2.0:
                walls = list(self.state.walls)
                walls.append(Segment(start, end))
                self.state.set_walls(walls)
            return
        super().mouseReleaseEvent(ev)


class TrackBuilderPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvas = _DrawableCanvas(state)
        splitter.addWidget(self.canvas)

        right_inner = QWidget()
        v = QVBoxLayout(right_inner)
        v.addWidget(self._build_grid_group())
        v.addWidget(self._build_file_group())
        v.addWidget(self._build_info_group())
        v.addStretch(1)
        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setWidget(right_inner)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1000, 400])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

        state.world_changed.connect(self._update_info)
        self._update_info()

    def _build_grid_group(self) -> QGroupBox:
        g = QGroupBox("Grid & Snap")
        lay = QVBoxLayout(g)
        self.show_grid_cb = QCheckBox("Show grid")
        self.show_grid_cb.setChecked(True)
        self.snap_cb = QCheckBox("Snap to grid")
        self.snap_cb.setChecked(True)
        lay.addWidget(self.show_grid_cb)
        lay.addWidget(self.snap_cb)
        row = QHBoxLayout()
        row.addWidget(QLabel("Cell size (cm):"))
        self.cell = QDoubleSpinBox()
        self.cell.setRange(0.5, 100.0)
        self.cell.setSingleStep(0.5)
        self.cell.setValue(10.0)
        row.addWidget(self.cell)
        lay.addLayout(row)
        self.show_grid_cb.toggled.connect(self._apply_grid)
        self.snap_cb.toggled.connect(self._apply_grid)
        self.cell.valueChanged.connect(lambda _v: self._apply_grid())
        self._apply_grid()
        hint = QLabel("Click empty space and drag to draw a wall. Drag yellow "
                      "handles to move endpoints. Right-click a wall to delete.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        lay.addWidget(hint)
        return g

    def _build_file_group(self) -> QGroupBox:
        g = QGroupBox("Track File")
        lay = QVBoxLayout(g)
        row1 = QHBoxLayout()
        btn_save = QPushButton("Save…")
        btn_load = QPushButton("Load…")
        btn_reset = QPushButton("Reset to Default")
        btn_clear = QPushButton("Clear All")
        btn_save.clicked.connect(self._save)
        btn_load.clicked.connect(self._load)
        btn_reset.clicked.connect(self.state.reset_world)
        btn_clear.clicked.connect(lambda: self.state.set_walls([]))
        row1.addWidget(btn_save)
        row1.addWidget(btn_load)
        lay.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(btn_reset)
        row2.addWidget(btn_clear)
        lay.addLayout(row2)
        return g

    def _build_info_group(self) -> QGroupBox:
        g = QGroupBox("Track Info")
        lay = QVBoxLayout(g)
        self.info = QLabel()
        self.info.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        lay.addWidget(self.info)
        return g

    def _apply_grid(self) -> None:
        self.canvas.set_grid(self.cell.value(),
                             self.snap_cb.isChecked(),
                             show=self.show_grid_cb.isChecked())

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Track",
                                              "track.yaml", "YAML (*.yaml *.yml)")
        if not path:
            return
        save_track(self.state.walls, path)

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Track",
                                              "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            walls = load_track(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self.state.set_walls(walls)

    def _update_info(self) -> None:
        walls = self.state.walls
        b = self.state.world_bounds
        total_len = sum(((s.b.x - s.a.x) ** 2 + (s.b.y - s.a.y) ** 2) ** 0.5
                        for s in walls)
        self.info.setText(
            f"walls   : {len(walls)}\n"
            f"bounds  : x [{b['min_x']:.1f}, {b['max_x']:.1f}] cm\n"
            f"          y [{b['min_y']:.1f}, {b['max_y']:.1f}] cm\n"
            f"perim.  : {total_len:.1f} cm"
        )
