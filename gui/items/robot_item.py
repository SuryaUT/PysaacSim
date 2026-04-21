"""QGraphicsItem that draws a robot (chassis, wheels, sensor placements).

Scene is Y-flipped (view applies scale(1,-1)) so (+x forward, +y left) in
world coords maps naturally. Robot is drawn at origin then positioned +
rotated by the scene transform.
"""
from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen, QPainterPath, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem

from ..app_state import RobotDims, RobotSpec
from ...sim.calibration import SensorCalibration


class RobotItem(QGraphicsItem):
    """Renders a single robot. Flags make it movable + rotatable via the
    parent QGraphicsView's mouse handling."""

    def __init__(self, spec: RobotSpec, dims: RobotDims,
                 calibration: SensorCalibration,
                 show_sensors: bool = True,
                 movable: bool = True):
        super().__init__()
        self.spec = spec
        self.dims = dims
        self.calibration = calibration
        self.show_sensors = show_sensors
        self._sensor_rays: list[tuple[float, float, float, float]] = []  # ox, oy, ex, ey in body frame
        self._build_sensor_rays()

        flags = QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        if movable:
            flags |= QGraphicsItem.GraphicsItemFlag.ItemIsMovable
        self.setFlags(flags)
        self.setPos(spec.x, spec.y)
        self.setRotation(math.degrees(spec.theta))

    # -- sync helpers -------------------------------------------------------

    def set_from_spec(self, spec: RobotSpec) -> None:
        self.spec = spec
        self.setPos(spec.x, spec.y)
        self.setRotation(math.degrees(spec.theta))
        self.update()

    def refresh_dims(self, dims: RobotDims, calibration: SensorCalibration) -> None:
        self.prepareGeometryChange()
        self.dims = dims
        self.calibration = calibration
        self._build_sensor_rays()
        self.update()

    def _build_sensor_rays(self) -> None:
        """Pre-compute short sensor ray stubs in body frame for visualization."""
        rays = []
        length = 25.0  # cm — display only
        for p in list(self.calibration.lidar_placements) + list(self.calibration.ir_placements):
            ex = p.x + length * math.cos(p.theta)
            ey = p.y + length * math.sin(p.theta)
            rays.append((p.x, p.y, ex, ey))
        self._sensor_rays = rays

    # -- paint --------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        L = self.dims.chassis_length_cm
        W = self.dims.chassis_width_cm
        # Include sensor rays in bounds.
        pad = 30.0
        half = max(L, W) / 2 + pad
        return QRectF(-half, -half, 2 * half, 2 * half)

    def paint(self, painter, option, widget=None) -> None:
        L = self.dims.chassis_length_cm
        W = self.dims.chassis_width_cm

        body_color = QColor(self.spec.color)
        body_color.setAlpha(210)
        painter.setPen(QPen(QColor("#111"), 0.4))
        painter.setBrush(QBrush(body_color))
        painter.drawRoundedRect(QRectF(-L / 2, -W / 2, L, W), 2.0, 2.0)

        # Heading arrow (forward +x).
        arrow = QPolygonF([
            QPointF(L / 2 - 3.5, -2.0),
            QPointF(L / 2,         0.0),
            QPointF(L / 2 - 3.5,  2.0),
        ])
        painter.setBrush(QBrush(QColor("#fff")))
        painter.drawPolygon(arrow)

        # Wheels (rough): 2 rear + 2 front at axle positions.
        from ...sim import constants as C
        front_x = L / 2 - C.FRONT_AXLE_FROM_FRONT_CM
        rear_x = -L / 2 + C.REAR_AXLE_FROM_REAR_CM
        wheel_l = self.dims.front_wheel_diam_cm
        wheel_w = 2.5
        painter.setBrush(QBrush(QColor("#222")))
        for ax, ay in [(front_x,  W / 2), (front_x, -W / 2),
                       (rear_x,   W / 2), (rear_x, -W / 2)]:
            painter.drawRect(QRectF(ax - wheel_l / 2, ay - wheel_w / 2, wheel_l, wheel_w))

        if self.show_sensors:
            pen_ir = QPen(QColor(255, 180, 50, 180), 0.5)
            pen_lidar = QPen(QColor(120, 200, 255, 180), 0.5)
            pen_lidar.setStyle(Qt.PenStyle.DashLine)
            lid_n = len(self.calibration.lidar_placements)
            for i, (ox, oy, ex, ey) in enumerate(self._sensor_rays):
                painter.setPen(pen_lidar if i < lid_n else pen_ir)
                painter.drawLine(QPointF(ox, oy), QPointF(ex, ey))
                painter.setBrush(QBrush(QColor("#fff")))
                painter.setPen(QPen(QColor("#111"), 0.3))
                painter.drawEllipse(QPointF(ox, oy), 0.8, 0.8)

    # -- interaction --------------------------------------------------------

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            # notify app_state — handled by canvas on mouseRelease for throttling
            pass
        return super().itemChange(change, value)
