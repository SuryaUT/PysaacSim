"""Default oval track. 32 in (81.28 cm) lane width, 3.5 in (8.89 cm) wall
thickness — matches the ECE445M Lab 7 track. Outer + inner rounded rectangles,
12 wall segments each."""
from __future__ import annotations

from .geometry import Segment, Vec2


LANE = 81.28   # 32 in
INNER_W = 60.0
INNER_L = 300.0
OUTER_W = INNER_W + 2 * LANE   # 232.72
OUTER_L = INNER_L + 2 * LANE   # 472.72


def _rounded_rect(x0: float, y0: float, L: float, W: float, cut: float) -> list[Segment]:
    x1, y1 = x0 + L, y0 + W
    pts = [
        Vec2(x0 + cut, y0), Vec2(x1 - cut, y0),
        Vec2(x1, y0 + cut), Vec2(x1, y1 - cut),
        Vec2(x1 - cut, y1), Vec2(x0 + cut, y1),
        Vec2(x0, y1 - cut), Vec2(x0, y0 + cut),
    ]
    return [Segment(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def build_default_world() -> dict:
    outer = _rounded_rect(0, 0, OUTER_L, OUTER_W, 35)
    inner = _rounded_rect(LANE, LANE, INNER_L, INNER_W, 25)
    return {
        "walls": outer + inner,
        "wall_thickness_cm": 8.89,
        "bounds": {"min_x": -10, "min_y": -10, "max_x": OUTER_L + 10, "max_y": OUTER_W + 10},
    }


# Bottom straight, spawned near the wall-follower equilibrium: body center at
# y=20.6 cm puts the right IR (body y=-7.62) at world y=12.98 cm ≈ dist_ref=130
# target (13 cm true IR distance after firmware /2).
DEFAULT_SPAWN = {"x": OUTER_L * 0.25, "y": 20.6, "theta": 0.0}
