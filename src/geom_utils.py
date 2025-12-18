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
