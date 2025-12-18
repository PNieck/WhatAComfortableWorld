import re
from typing import List

import numpy as np

from .parsing_errors import *
from src.floor_plan import FloorPlan, FrontDoor, Room, RoomType


def _coords_from_sequence(seq: str):
    matches = re.findall(r"<Coord (\d+)>", seq)

    if len(matches) % 2 != 0:
        raise OddNumberOfCoordinatesError(f"Coordinates number ({len(matches)}) is not divisible by 2")

    coords = np.array(list(map(int, matches)), dtype=np.uint8)
    result = coords.reshape(-1, 2)
    
    return result


def boundary_from_sequence(seq: str) -> np.ndarray:
    match = re.search(r"<Bound>(<Coord \d+>)+", seq)
    if not match:
        raise NoBoundaryError()
    
    result = _coords_from_sequence(match.group())
    if result.shape[0] < 3:
        raise TooSmallNumberOfCoordinatesError(result.shape[0])
    
    return result


def door_from_sequence(seq: str) -> FrontDoor:
    match = re.search(r"<Door>(<Coord \d+>)+", seq)
    if not match:
        raise NoFrontDoorsError()
    
    coords = _coords_from_sequence(match.group())
    if coords.shape != (2, 2):
        raise CoordinatesNumberForFrontDoorError(f"Got {coords.size} coordinates, expected is 4")

    return FrontDoor(coords)


def _rooms_from_sequence(seq: str) -> List[Room]:
    regex = r"<Room \d+>(<Coord \d+>)+"
    result = []

    for match in re.finditer(regex, seq):
        room_type_match = re.match(r"<Room (\d+)>", match.group())
        assert(room_type_match)

        room_type = int(room_type_match.group(1))

        boundary = _coords_from_sequence(match.group())
        if boundary.shape[0] < 3:
            raise TooSmallNumberOfCoordinatesError(boundary.shape[0])

        result.append(Room(RoomType(room_type), boundary))

    if not result:
        raise NoRoomsError()

    return result


def from_sequence(seq: str, name: str ="") -> FloorPlan:
    boundary = boundary_from_sequence(seq)
    doors = door_from_sequence(seq)
    rooms = _rooms_from_sequence(seq)

    return FloorPlan(name, boundary, doors, rooms)