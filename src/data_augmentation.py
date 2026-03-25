"""Data augmentation utilities for floor plan transformations.

Provides functions for augmenting floor plan datasets through permutation,
symmetry operations and rotations.
"""

from src.floor_plan import FloorPlan, FrontDoor, Room

import random
from enum import Enum

import numpy as np


def permutate(floor_plan: FloorPlan) -> FloorPlan:
    """
    Created copy of floor plan with changed order of rooms
    
    :param floor_plan: floor plan to permutate rooms
    :type floor_plan: FloorPlan
    :return: Copy of passed flor plan with changed order of rooms
    :rtype: FloorPlan
    """\

    return FloorPlan(
        floor_plan.name + "_permuted",
        floor_plan.boundary,
        floor_plan.front_door,

        # Creating a permuted copy of rooms list
        random.sample(floor_plan.rooms, len(floor_plan.rooms))
    )


class SymmetryType(Enum):
    """Possible symmetry types"""

    X = 0   # Symmetry along vertical axes
    Y = 1   # Symmetry along horizontal axes
    XY = 2  # Symmetry along both axes


def _coord_symmetry(coord: np.ndarray) -> np.ndarray:
    """Reflects symmetrically 1D numpy array of coordinates"""

    if coord.dtype == np.uint8:
        coord = coord.astype(np.int32)

    result = (FloorPlan.MAX_COORDINATE + 1) - coord

    return result


def _arr_sym(arr: np.ndarray, type: SymmetryType):
    """
    Flips polygon vertices along vertical (y), horizontal (x) or both axes
    
    :param arr: Numpy array of shape (n, 2) with polygon vertices
    :type arr: np.ndarray
    :param type: Type of symmetry
    :type type: SymmetryType
    """

    result = np.empty_like(arr)

    match type:
        case SymmetryType.X:
            result[:, 0] = _coord_symmetry(arr[:, 0])
            result[:, 1] = arr[:, 1]

        case SymmetryType.Y:
            result[:, 0] = arr[:, 0]
            result[:, 1] = _coord_symmetry(arr[:, 1])

        case SymmetryType.XY:
            result[:, 0] = _coord_symmetry(arr[:, 0])
            result[:, 1] = _coord_symmetry(arr[:, 1])

        case _:
            raise Exception(f"Unknown symmetry type {type}")
    
    return result


def symmetry(floor_plan: FloorPlan, type: SymmetryType) -> FloorPlan:
    """
    Flips floor plan along vertical (y), horizontal (x) or both axes
    
    :param floor_plan: floor plan to be flipped
    :type floor_plan: FloorPlan
    :param type: Type of symmetry
    :type type: SymmetryType
    :return: Flipped floor plan
    :rtype: FloorPlan
    """

    boundary = _arr_sym(floor_plan.boundary, type)
    door = FrontDoor(_arr_sym(floor_plan.front_door.corners, type))

    rooms = [None] * floor_plan.room_cnt

    for i, room in enumerate(floor_plan.rooms):
        room_boundary = _arr_sym(room.boundary, type)

        rooms[i] = Room(room.type, room_boundary)

    return FloorPlan(
        floor_plan.name + "_sym",
        boundary,
        door,
        rooms
    )


class RotationAngle(Enum):
    """
    Possible angles to perform rotation
    """

    _90deg = 0
    _180deg = 1
    _270deg = 2


def _arr_rot(arr: np.ndarray, angle: RotationAngle) -> np.ndarray:
    """
    Rotates polygon vertices by specified angle in clock-wise manner
    
    :param arr: Numpy array of shape (n, 2) with polygon vertices
    :type arr: np.ndarray
    :param angle: Angle by which array has to be rotated
    :type angle: RotationAngle
    :return: Rotated version of vertices
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """

    result = np.empty_like(arr)

    if arr.dtype == np.uint8:
        arr = arr.astype(np.uint16)

    match angle:
        case RotationAngle._90deg:
            result[:, 0] = (FloorPlan.MAX_COORDINATE + 1) - arr[:, 1]
            result[:, 1] = arr[:, 0]

        case RotationAngle._180deg:
            result[:, 0] = (FloorPlan.MAX_COORDINATE + 1) - arr[:, 0]
            result[:, 1] = (FloorPlan.MAX_COORDINATE + 1) - arr[:, 1]

        case RotationAngle._270deg:
            result[:, 0] = arr[:, 1]
            result[:, 1] = (FloorPlan.MAX_COORDINATE + 1) - arr[:, 0]

        case _:
            raise Exception("Unknown angle")        

    return result


def rotation(floor_plan: FloorPlan, angle: RotationAngle) -> FloorPlan:
    """
    Rotates floor plan by specified angle in clock-wise manner
    
    :param floor_plan: Floor plan, which is going to be rotated
    :type floor_plan: FloorPlan
    :param angle: Angle by which floor plan is going to be rotated
    :type angle: RotationAngle
    :return: The the floor plan, which is a rotated version of one specified in the arguments
    """

    boundary = _arr_rot(floor_plan.boundary, angle)
    door = FrontDoor(_arr_rot(floor_plan.front_door.corners, angle))

    rooms = [None] * floor_plan.room_cnt

    for i, room in enumerate(floor_plan.rooms):
        room_boundary = _arr_rot(room.boundary, angle)

        rooms[i] = Room(room.type, room_boundary)

    return FloorPlan(
        floor_plan.name + "_rot",
        boundary,
        door,
        rooms
    )