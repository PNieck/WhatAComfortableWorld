from src.floor_plan import FloorPlan, FrontDoor, Room

import random
from enum import Enum

import numpy as np


def permutate(floor_plan: FloorPlan) -> FloorPlan:
    return FloorPlan(
        floor_plan.name + "_permuted",
        floor_plan.boundary,
        floor_plan.front_door,

        # Creating a permuted copy of rooms list
        random.sample(floor_plan.rooms, len(floor_plan.rooms))
    )


class SymmetryType(Enum):
    X = 0
    Y = 1
    XY = 2


def _coord_symmetry(coord):
    return (FloorPlan.MAX_COORDINATE + 1) - coord


def _arr_sym(arr: np.ndarray, type: SymmetryType):
    result = np.empty_like(arr)

    if type == SymmetryType.X:
        result[:, 0] = _coord_symmetry(arr[:, 0])
        result[:, 1] = arr[:, 1]

    elif type == SymmetryType.Y:
        result[:, 0] = arr[:, 0]
        result[:, 1] = _coord_symmetry(arr[:, 1])

    elif type == SymmetryType.XY:
        result[:, 0] = _coord_symmetry(arr[:, 0])
        result[:, 1] = _coord_symmetry(arr[:, 1])

    else:
        raise Exception("Unknown symmetry type")
    
    return result


def symmetry(floor_plan: FloorPlan, type: SymmetryType) -> FloorPlan:
    boundary = _arr_sym(floor_plan.boundary, type)
    door = FrontDoor(_arr_sym(floor_plan.front_door.corners, type))

    rooms = [None] * floor_plan.rooms_cnt

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
    _90deg = 0
    _180deg = 1
    _270deg = 2


def _arr_rot(arr: np.ndarray, angle: RotationAngle):
    result = np.empty_like(arr)

    if angle == RotationAngle._90deg:
        result[:, 0] = (FloorPlan.MAX_COORDINATE + 1) - arr[:, 1]
        result[:, 1] = arr[:, 0]

    elif angle == RotationAngle._180deg:
        result[:, 0] = (FloorPlan.MAX_COORDINATE + 1) - arr[:, 0]
        result[:, 1] = (FloorPlan.MAX_COORDINATE + 1) - arr[:, 1]

    elif angle == RotationAngle._270deg:
        result[:, 0] = arr[:, 1]
        result[:, 1] = (FloorPlan.MAX_COORDINATE + 1) - arr[:, 0]

    else:
        raise Exception("Unknown angle")

    return result


def rotation(floor_plan: FloorPlan, angle: RotationAngle):
    boundary = _arr_rot(floor_plan.boundary, angle)
    door = FrontDoor(_arr_rot(floor_plan.front_door.corners, angle))

    rooms = [None] * floor_plan.rooms_cnt

    for i, room in enumerate(floor_plan.rooms):
        room_boundary = _arr_rot(room.boundary, angle)

        rooms[i] = Room(room.type, room_boundary)

    return FloorPlan(
        floor_plan.name + "_rot",
        boundary,
        door,
        rooms
    )