"""Rotation ring overlay shown around the currently selected robot.

The user drags anywhere on the ring (or its grip arrow) and the angle from
the robot center to the cursor becomes the new heading. Emits updates via
the supplied callback.
"""
from __future__ import annotations

import math
from typing import Callable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen, QPainterPath, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem


class RotationRing(QGraphicsItem):
    """Floating overlay — positioned by the canvas at the robot's center.

    Interactive: clicking/dragging within the ring annulus sets the robot's
    theta to the angle from the center to the cursor. Non-movable itself
    (the canvas sets its pos to match the robot's pos each frame).
    """
    def __init__(self, radius_cm: float,
                 on_angle: Callable[[float], None],
                 on_release: Callable[[], None]):
        super().__init__()
        self._radius = radius_cm
        # Thicker so it's easy to grab with a trackpad.
        self._thickness = max(3.5, radius_cm * 0.22)
        self._on_angle = on_angle
        self._on_release = on_release
        self.setAcceptHoverEvents(True)
        self.setZValue(25)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    # -- visuals -----------------------------------------------------------

    def boundingRect(self) -> QRectF:
        r = self._radius + self._thickness * 2 + 2.0
        return QRectF(-r, -r, 2 * r, 2 * r)

    def shape(self) -> QPainterPath:
        # Hit-test only within the annulus so clicks in the empty center
        # (or outside the outer edge) pass through to whatever is underneath
        # instead of being swallowed.
        inner = max(0.0, self._radius - self._thickness)
        outer = self._radius + self._thickness * 1.2
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        path.addEllipse(QPointF(0, 0), outer, outer)
        path.addEllipse(QPointF(0, 0), inner, inner)
        # Include the grip-arrow tip area so it is always grabbable.
        tip_r = self._radius + self._thickness * 1.6
        path.addEllipse(QPointF(self._radius, 0), self._thickness, self._thickness)
        _ = tip_r
        return path

    def paint(self, painter, option, widget=None):
        r = self._radius
        t = self._thickness
        # Annulus
        pen = QPen(QColor(33, 150, 243, 210), t)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(0, 0), r, r)
        # Grip arrow at +x (local frame — caller rotates the item so this
        # points along the robot heading).
        tip = QPointF(r + t * 1.2, 0.0)
        base_l = QPointF(r - t * 0.4, -t * 1.1)
        base_r = QPointF(r - t * 0.4, +t * 1.1)
        tri = QPolygonF([tip, base_l, base_r])
        painter.setBrush(QBrush(QColor(33, 150, 243)))
        painter.setPen(QPen(QColor("#0d47a1"), 0.4))
        painter.drawPolygon(tri)

    # -- mouse -------------------------------------------------------------

    def _angle_from_event_scene_pos(self, scene_pos: QPointF) -> float:
        center = self.scenePos()
        dx = scene_pos.x() - center.x()
        dy = scene_pos.y() - center.y()
        if dx * dx + dy * dy < 1e-6:
            return 0.0
        return math.atan2(dy, dx)

    def mousePressEvent(self, ev):
        # shape() already restricts hits to the annulus — just accept.
        self._on_angle(self._angle_from_event_scene_pos(ev.scenePos()))
        ev.accept()

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self._on_angle(self._angle_from_event_scene_pos(ev.scenePos()))
            ev.accept()

    def mouseReleaseEvent(self, ev):
        self._on_release()
        ev.accept()
