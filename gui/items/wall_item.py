"""Wall segment QGraphicsItem with draggable endpoints (for Track Builder)."""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
)

WALL_THICKNESS_CM = 8.89  # matches world.py default; can be overridden per-canvas

from ...sim.geometry import Segment, Vec2


ENDPOINT_RADIUS = 2.5  # cm — visual


class _Endpoint(QGraphicsEllipseItem):
    """Small draggable handle on a wall endpoint. Calls back into parent on move."""
    def __init__(self, idx: int, wall: "WallItem"):
        r = ENDPOINT_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r, parent=wall)
        self.idx = idx
        self._wall = wall
        self.setBrush(QBrush(QColor("#ffeb3b")))
        self.setPen(QPen(QColor("#111"), 0.3))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        self.setZValue(10)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            new_pos: QPointF = value
            new_pos = self._wall.snap_point(new_pos)
            self._wall.update_endpoint(self.idx, new_pos)
            return new_pos
        return super().itemChange(change, value)


class WallItem(QGraphicsLineItem):
    """A single wall segment + its two endpoint handles."""

    def __init__(self, a: Vec2, b: Vec2,
                 snap_fn: Optional[Callable[[QPointF], QPointF]] = None,
                 editable: bool = True,
                 thickness_cm: float = 8.89):
        super().__init__(a.x, a.y, b.x, b.y)
        # Pen width in world cm matches real wall thickness (not cosmetic so it
        # scales with zoom and shows real girth on the track).
        pen = QPen(QColor("#333"), thickness_cm)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self.setPen(pen)
        self._snap = snap_fn or (lambda p: p)
        self._a = Vec2(a.x, a.y)
        self._b = Vec2(b.x, b.y)
        self._editable = editable
        self._on_changed: Optional[Callable[[], None]] = None
        self._endpoints: list[_Endpoint] = []
        if editable:
            ea = _Endpoint(0, self); ea.setPos(a.x, a.y)
            eb = _Endpoint(1, self); eb.setPos(b.x, b.y)
            self._endpoints = [ea, eb]

    # -- API ----------------------------------------------------------------

    def segment(self) -> Segment:
        return Segment(Vec2(self._a.x, self._a.y), Vec2(self._b.x, self._b.y))

    def set_changed_callback(self, cb: Callable[[], None]) -> None:
        self._on_changed = cb

    def snap_point(self, p: QPointF) -> QPointF:
        return self._snap(p)

    def set_snap_fn(self, snap_fn: Callable[[QPointF], QPointF]) -> None:
        self._snap = snap_fn

    def update_endpoint(self, idx: int, pos: QPointF) -> None:
        if idx == 0:
            self._a = Vec2(pos.x(), pos.y())
        else:
            self._b = Vec2(pos.x(), pos.y())
        self.setLine(self._a.x, self._a.y, self._b.x, self._b.y)
        if self._on_changed is not None:
            self._on_changed()

    # -- right-click delete is handled by canvas via context menu -----------
