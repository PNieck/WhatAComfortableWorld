from enum import Enum

import shapely


class TurnType(Enum):
    Left = 0
    Right = 1
    Straight = 2


def turn_type(p1, p2, p3) -> TurnType:
    """Determine the turn type formed by three points.

    Args:
        p1: First point as a array-like (x, y).
        p2: Second point as a array-like (x, y).
        p3: Third point as a array-like (x, y).

    Returns:
        A TurnType object.
    """
    cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    if cross_product > 0:
        return TurnType.Left
    elif cross_product < 0:
        return TurnType.Right
    else:
        return TurnType.Straight


class LineType(Enum):
    Horizontal = 0
    Vertical = 1
    Diagonal = 2


def line_type(x1, y1, x2, y2) -> LineType:
    if y1 == y2:
        return LineType.Horizontal
    
    if x1 == x2:
        return LineType.Vertical
    
    return LineType.Diagonal


def create_line(p1, p2):
    if p1 == p2:
        return shapely.Point(p1)
    
    return shapely.LineString([p1, p2])


def line_strings_form_linear_ring(ring: shapely.LinearRing) -> list[shapely.LineString]:
    return [
        shapely.LineString((a, b))
        for a, b in zip(ring.coords[:-1], ring.coords[1:])
    ]


def _is_collinear(p0, p1, p2, eps=1e-12):
    """
    Return True if p0, p1, p2 are collinear within numerical tolerance eps.
    Works for 2D or 3D points.
    """
    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    # Cross product z-component of (p1-p0) x (p2-p1)
    cp = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
    return abs(cp) <= eps


def remove_collinear_points(line: shapely.LineString, eps: float = 1e-12) -> shapely.LineString:
    """
    Return a new LineString with interior points removed when they are
    collinear with their neighbors (within eps).
    """
    coords = list(line.coords)
    n = len(coords)

    if n <= 2:
        # Nothing to simplify
        return line

    new_coords = [coords[0]]

    for i in range(1, n - 1):
        p_prev = coords[i - 1]
        p_cur = coords[i]
        p_next = coords[i + 1]

        if not _is_collinear(p_prev, p_cur, p_next, eps=eps):
            new_coords.append(p_cur)

    new_coords.append(coords[-1])
    return shapely.LineString(new_coords)


def remove_collinear_points_ring(ring: shapely.LinearRing, eps: float = 1e-12) -> shapely.LinearRing:
    """
    Remove vertices from a LinearRing that are collinear with their neighbors.
    The result is closed and has at least 3 distinct vertices (if possible).
    """
    coords = list(ring.coords)

    # coords in a LinearRing are already closed: [..., first]
    # work on the unique part
    unique = coords[:-1]
    n = len(unique)

    if n <= 3:
        # cannot drop anything without breaking validity
        return ring

    keep = [False] * n

    # always keep all vertices first, then selectively drop collinear ones
    # but ensure we never go below 3 kept vertices
    # first pass: mark non-collinear vertices
    for i in range(n):
        p_prev = unique[(i - 1) % n]
        p_cur  = unique[i]
        p_next = unique[(i + 1) % n]
        if not _is_collinear(p_prev, p_cur, p_next, eps=eps):
            keep[i] = True

    # guarantee at least 3 kept vertices
    if sum(keep) < 3:
        # fall back to original ring
        return ring

    new_unique = [p for p, k in zip(unique, keep) if k]

    # close the ring
    new_coords = new_unique + [new_unique[0]]
    return shapely.LinearRing(new_coords)

