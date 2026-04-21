"""2D geometry primitives. All distances in cm."""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple


class Vec2(NamedTuple):
    x: float
    y: float


class Segment(NamedTuple):
    a: Vec2
    b: Vec2


def body_to_world(pose: Tuple[float, float, float], bx: float, by: float) -> Vec2:
    """Transform a body-frame point (bx, by) by (x, y, theta) to world frame."""
    c = math.cos(pose[2])
    s = math.sin(pose[2])
    return Vec2(pose[0] + bx * c - by * s, pose[1] + bx * s + by * c)


def ray_hits_segment(
    origin: Vec2, dir: Vec2, max_dist: float, seg_a: Vec2, seg_b: Vec2
) -> Optional[Tuple[float, Vec2, Vec2]]:
    """Ray vs segment. Returns (distance_along_ray, hit_point, normal) or None.
    `dir` must be a unit vector."""
    sx, sy = seg_b.x - seg_a.x, seg_b.y - seg_a.y
    denom = dir.x * sy - dir.y * sx
    if abs(denom) < 1e-12:
        return None
    oax, oay = seg_a.x - origin.x, seg_a.y - origin.y
    t = (oax * sy - oay * sx) / denom        # distance along ray
    u = (oax * dir.y - oay * dir.x) / denom  # parameter along segment
    if t < 0 or t > max_dist or u < 0 or u > 1:
        return None
    seg_len = math.hypot(sx, sy)
    normal = Vec2(0.0, 0.0) if seg_len < 1e-9 else Vec2(-sy / seg_len, sx / seg_len)
    point = Vec2(origin.x + dir.x * t, origin.y + dir.y * t)
    return (t, point, normal)


def cast_ray(
    walls: list[Segment], origin: Vec2, dir: Vec2, max_dist: float
) -> Optional[Tuple[float, Vec2, Vec2]]:
    """Nearest ray hit across all walls."""
    best: Optional[Tuple[float, Vec2, Vec2]] = None
    for w in walls:
        h = ray_hits_segment(origin, dir, max_dist, w.a, w.b)
        if h is not None and (best is None or h[0] < best[0]):
            best = h
    return best


def seg_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> Optional[Vec2]:
    """Segment-segment intersection point, or None."""
    rx, ry = b.x - a.x, b.y - a.y
    sx, sy = d.x - c.x, d.y - c.y
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return None
    acx, acy = c.x - a.x, c.y - a.y
    t = (acx * sy - acy * sx) / denom
    u = (acx * ry - acy * rx) / denom
    if t < 0 or t > 1 or u < 0 or u > 1:
        return None
    return Vec2(a.x + rx * t, a.y + ry * t)


def chassis_segments(
    pose: Tuple[float, float, float], length_cm: float, width_cm: float
) -> list[Segment]:
    """Rectangular chassis footprint as four world-frame segments."""
    hl = length_cm / 2.0
    hw = width_cm / 2.0
    corners = [
        body_to_world(pose, +hl, +hw),
        body_to_world(pose, +hl, -hw),
        body_to_world(pose, -hl, -hw),
        body_to_world(pose, -hl, +hw),
    ]
    return [
        Segment(corners[0], corners[1]),
        Segment(corners[1], corners[2]),
        Segment(corners[2], corners[3]),
        Segment(corners[3], corners[0]),
    ]
