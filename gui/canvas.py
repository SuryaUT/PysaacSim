"""TrackCanvas — QGraphicsView rendering the track, robots, and sensor rays.

World coords are in cm with +y up. Qt's scene Y axis is down; we apply
scale(1, -1) on the view transform and negate Y when translating mouse
coordinates back to the scene.

The canvas has several `mode`s:
  - "sim":       robots are draggable (Ctrl+click sets rotation), walls are not.
  - "build":     walls are draggable with endpoint handles, robots are hidden.
  - "robot":     single robot centered, no walls — used by Robot Builder.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from PyQt6.QtCore import QEvent, QLineF, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QKeyEvent, QMouseEvent, QPainter, QPen, QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsLineItem, QGraphicsScene, QGraphicsView, QMenu,
)

from ..sim.geometry import Segment, Vec2
from .app_state import AppState, RobotSpec
from .items.robot_item import RobotItem
from .items.rotation_ring import RotationRing
from .items.wall_item import WallItem


class TrackCanvas(QGraphicsView):
    """Interactive canvas over AppState. Rebuilds scene on state signals."""

    robot_clicked = pyqtSignal(int)          # robot_id selected
    robot_moved = pyqtSignal(int, float, float, float)   # id, x, y, theta (after drag)
    empty_placed = pyqtSignal(float, float, float)  # world x, y, theta (heading)
    rotate_selected_requested = pyqtSignal(int)    # +1 ccw / -1 cw (for R / Shift+R)
    robot_theta_changed = pyqtSignal(int, float)   # rotation-ring drag (id, theta)

    def __init__(self, state: AppState, mode: str = "sim", parent=None):
        super().__init__(parent)
        self.state = state
        self.mode = mode
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Y-flip so world +y is up on screen.
        self.scale(1.0, -1.0)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QBrush(QColor("#f6f6f6")))
        self.setMouseTracking(True)

        self._grid_cm: float = 10.0
        self._snap_enabled: bool = True
        self._show_grid: bool = (mode == "build")
        self._robot_items: dict[int, RobotItem] = {}
        self._wall_items: list[WallItem] = []
        self._selected_robot_id: Optional[int] = None
        self._rotation_ring: Optional[RotationRing] = None
        # drag-to-place-with-heading state (sim mode)
        self._placing_from: Optional[QPointF] = None
        self._place_preview: Optional[QGraphicsLineItem] = None
        # Accept keyboard focus so R / Shift+R work after clicking.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        state.world_changed.connect(self._rebuild_walls)
        state.robots_changed.connect(self._rebuild_robots)
        state.calibration_changed.connect(self._refresh_robot_dims)
        state.dims_changed.connect(self._refresh_robot_dims)

        self._build_scene()
        self.fit()

    # ---- public API -------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._rebuild_walls()
        self._rebuild_robots()

    def set_grid(self, cm: float, enabled: bool, show: Optional[bool] = None) -> None:
        self._grid_cm = max(0.1, cm)
        self._snap_enabled = enabled
        if show is not None:
            self._show_grid = show
        for w in self._wall_items:
            w.set_snap_fn(self._snap_fn())
        self.viewport().update()

    def fit(self) -> None:
        b = self.state.world_bounds
        rect = QRectF(b["min_x"], b["min_y"],
                      b["max_x"] - b["min_x"], b["max_y"] - b["min_y"])
        self._scene.setSceneRect(rect)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def highlight_robot(self, robot_id: Optional[int]) -> None:
        self._selected_robot_id = robot_id
        for rid, item in self._robot_items.items():
            item.setOpacity(1.0 if rid == robot_id or robot_id is None else 0.55)
        self._update_rotation_ring()

    def _update_rotation_ring(self) -> None:
        """Show ring around selected robot (sim mode only), remove otherwise."""
        # Remove old ring.
        if self._rotation_ring is not None:
            self._scene.removeItem(self._rotation_ring)
            self._rotation_ring = None
        if self.mode != "sim" or self._selected_robot_id is None:
            return
        item = self._robot_items.get(self._selected_robot_id)
        if item is None:
            return
        # Ring radius based on chassis size.
        dims = self.state.dims
        r = max(dims.chassis_length_cm, dims.chassis_width_cm) * 0.75

        rid = self._selected_robot_id
        def on_angle(theta: float) -> None:
            self.robot_theta_changed.emit(rid, theta)
        def on_release() -> None:
            pass
        ring = RotationRing(r, on_angle, on_release)
        self._scene.addItem(ring)
        ring.setPos(item.pos())
        ring.setRotation(item.rotation())
        self._rotation_ring = ring

    # ---- scene construction -----------------------------------------------

    def _build_scene(self) -> None:
        self._rebuild_walls()
        self._rebuild_robots()

    def _rebuild_walls(self) -> None:
        for w in self._wall_items:
            self._scene.removeItem(w)
        self._wall_items = []
        editable = self.mode == "build"
        thickness = float(self.state._world.get("wall_thickness_cm", 8.89))
        for seg in self.state.walls:
            item = WallItem(seg.a, seg.b, snap_fn=self._snap_fn(),
                            editable=editable, thickness_cm=thickness)
            item.set_changed_callback(self._walls_updated_from_items)
            self._scene.addItem(item)
            self._wall_items.append(item)
        self.viewport().update()

    def _rebuild_robots(self) -> None:
        for item in self._robot_items.values():
            self._scene.removeItem(item)
        self._robot_items = {}
        # Any previous ring has stale item refs after a rebuild.
        if self._rotation_ring is not None:
            self._scene.removeItem(self._rotation_ring)
            self._rotation_ring = None
        if self.mode == "build":
            return
        for spec in self.state.robots:
            movable = self.mode == "sim"
            item = RobotItem(spec, self.state.dims, self.state.calibration,
                             show_sensors=True, movable=movable)
            item.setZValue(5)
            self._scene.addItem(item)
            self._robot_items[spec.id] = item
        self.highlight_robot(self._selected_robot_id)

    def sync_ring_to_selected(self) -> None:
        """Called each sim frame so the ring follows the robot during play."""
        if self._rotation_ring is None or self._selected_robot_id is None:
            return
        item = self._robot_items.get(self._selected_robot_id)
        if item is None:
            return
        self._rotation_ring.setPos(item.pos())
        self._rotation_ring.setRotation(item.rotation())

    def _refresh_robot_dims(self) -> None:
        for item in self._robot_items.values():
            item.refresh_dims(self.state.dims, self.state.calibration)

    def _walls_updated_from_items(self) -> None:
        """One of the wall endpoints moved — push back into app_state."""
        segs = [w.segment() for w in self._wall_items]
        self.state._world["walls"] = segs  # direct to avoid bounds recompute during drag
        # defer world_changed until drag-release for perf; endpoints still update visually

    def commit_walls(self) -> None:
        """Called by pages (e.g., after mouse release) to re-emit world_changed
        and recompute bounds."""
        segs = [w.segment() for w in self._wall_items]
        self.state.set_walls(segs)

    # ---- mouse / snapping -------------------------------------------------

    def _snap_fn(self) -> Callable[[QPointF], QPointF]:
        g = self._grid_cm
        enabled = self._snap_enabled
        if not enabled:
            return lambda p: p
        def snap(p: QPointF) -> QPointF:
            return QPointF(round(p.x() / g) * g, round(p.y() / g) * g)
        return snap

    def _scene_pos_from(self, ev: QMouseEvent) -> QPointF:
        return self.mapToScene(ev.pos())

    def wheelEvent(self, ev: QWheelEvent) -> None:
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        scene_pos = self._scene_pos_from(ev)
        if self._snap_enabled:
            scene_pos = self._snap_fn()(scene_pos)

        if ev.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(ev.pos())
            on_robot = isinstance(item, RobotItem)
            on_wall = isinstance(item, WallItem)
            on_ring = isinstance(item, RotationRing)
            if on_ring:
                # Let the ring handle its own drag — don't treat as empty.
                super().mousePressEvent(ev)
                return
            if not on_robot and not on_wall:
                # Empty space: in sim mode, start drag-to-place w/ heading.
                if self.mode == "sim":
                    self._placing_from = scene_pos
                    return
                super().mousePressEvent(ev)
                return
            if on_robot:
                self.robot_clicked.emit(item.spec.id)
            super().mousePressEvent(ev)
            return

        if ev.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(ev)
            return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._placing_from is not None:
            pos = self._scene_pos_from(ev)
            if self._place_preview is None:
                line = QGraphicsLineItem()
                pen = QPen(QColor("#2196f3"), 0.6)
                pen.setCosmetic(True)
                pen.setStyle(Qt.PenStyle.DashLine)
                line.setPen(pen)
                line.setZValue(30)
                self._scene.addItem(line)
                self._place_preview = line
            self._place_preview.setLine(self._placing_from.x(),
                                        self._placing_from.y(),
                                        pos.x(), pos.y())
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        # Handle drag-to-place in sim mode.
        if ev.button() == Qt.MouseButton.LeftButton and self._placing_from is not None:
            start = self._placing_from
            end = self._scene_pos_from(ev)
            if self._place_preview is not None:
                self._scene.removeItem(self._place_preview)
                self._place_preview = None
            self._placing_from = None
            dx, dy = end.x() - start.x(), end.y() - start.y()
            if (dx * dx + dy * dy) > 4.0:   # >2cm drag → use heading
                theta = math.atan2(dy, dx)
            else:
                theta = 0.0
            sx, sy = start.x(), start.y()
            if self._snap_enabled:
                snapped = self._snap_fn()(start)
                sx, sy = snapped.x(), snapped.y()
            self.empty_placed.emit(sx, sy, theta)
            return

        super().mouseReleaseEvent(ev)
        # Commit robot drags back into app state.
        for rid, item in self._robot_items.items():
            pos = item.pos()
            theta_deg = item.rotation()
            theta_rad = math.radians(theta_deg)
            spec = next((r for r in self.state.robots if r.id == rid), None)
            if spec is None:
                continue
            if (abs(pos.x() - spec.x) > 1e-3 or abs(pos.y() - spec.y) > 1e-3
                    or abs(theta_rad - spec.theta) > 1e-4):
                self.robot_moved.emit(rid, pos.x(), pos.y(), theta_rad)
        # Commit any wall drags in build mode.
        if self.mode == "build":
            self.commit_walls()

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        if ev.key() == Qt.Key.Key_R and self.mode == "sim":
            direction = +1 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else -1
            # Convention: R = clockwise (−), Shift+R = counter-clockwise (+).
            self.rotate_selected_requested.emit(direction)
            return
        super().keyPressEvent(ev)

    def _show_context_menu(self, ev: QMouseEvent) -> None:
        item = self.itemAt(ev.pos())
        menu = QMenu(self)

        if isinstance(item, RobotItem):
            act_rotate_l = menu.addAction("Rotate -15°")
            act_rotate_r = menu.addAction("Rotate +15°")
            act_delete = menu.addAction("Delete Robot")
            chosen = menu.exec(ev.globalPosition().toPoint())
            if chosen == act_delete:
                self.state.remove_robot(item.spec.id)
            elif chosen == act_rotate_l:
                self.state.update_robot(item.spec.id, theta=item.spec.theta - math.radians(15))
            elif chosen == act_rotate_r:
                self.state.update_robot(item.spec.id, theta=item.spec.theta + math.radians(15))
            return

        if isinstance(item, WallItem) and self.mode == "build":
            act_delete = menu.addAction("Delete Wall")
            chosen = menu.exec(ev.globalPosition().toPoint())
            if chosen == act_delete:
                self._scene.removeItem(item)
                self._wall_items.remove(item)
                self.commit_walls()
            return

    # ---- background grid --------------------------------------------------

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)
        if not self._show_grid:
            return
        g = self._grid_cm
        pen = QPen(QColor(0, 0, 0, 25))
        pen.setCosmetic(True)
        painter.setPen(pen)
        left = math.floor(rect.left() / g) * g
        top = math.floor(rect.top() / g) * g
        x = left
        while x <= rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += g
        y = top
        while y <= rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += g
