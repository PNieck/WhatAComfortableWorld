from typing import List
import itertools
import re

import tokens
from src.floor_plan import FloorPlan, FrontDoor, Room, RoomType

import numpy as np


def boundary_sequence(plan: FloorPlan) -> List[str]:
    result = [None] * (plan.boundary_len * 2 + 1)
    
    result[0] = tokens.BOUNDARY_TOKEN

    for i, corner in enumerate(plan.boundary):
        result[2*i+1] = tokens.coord_token(corner[0])
        result[2*i+2] = tokens.coord_token(corner[1])

    return result


def door_sequence(door: FrontDoor) -> List[str]:
    return [
        tokens.DOOR_TOKEN,
        tokens.coord_token(door.x1),
        tokens.coord_token(door.y1),
        tokens.coord_token(door.x2),
        tokens.coord_token(door.y2),
    ]


def room_sequence(room: Room) -> List[str]:
    result = [None] * (room.boundary_len * 2 + 1)

    result[0] = tokens.room_token(room.type.value)

    for i, corner in enumerate(room.boundary):
        result[2*i + 1] = tokens.coord_token(corner[0])
        result[2*i + 2] = tokens.coord_token(corner[1])

    return result


def to_sequence(plan: FloorPlan) -> List[str]:
    boundary_seq = boundary_sequence(plan)
    door_seq = door_sequence(plan.front_door)
    rooms_seqs = [room_sequence(room) for room in plan.rooms]

    room_seq = itertools.chain.from_iterable(rooms_seqs)

    return list(itertools.chain(boundary_seq, door_seq, room_seq))


def _coords_from_sequence(seq: str):
    matches = re.findall(r"<Coord (\d+)>", seq)

    if len(matches) % 2 != 0:
        raise ValueError("Invalid number of coordinates")
    
    coords = np.array(list(map(int, matches)), dtype=np.uint8)
    return coords.reshape(-1, 2)


def _boundary_from_sequence(seq: str) -> np.ndarray:
    match = re.search(r"<Bound>(<Coord \d+>)+", seq)
    if not match:
        raise ValueError("No boundary in sequence")
    
    return _coords_from_sequence(match.group())


def _door_from_sequence(seq: str) -> FrontDoor:
    match = re.search(r"<Door>(<Coord \d+>)+", seq)
    if not match:
        raise ValueError("No front door in sequence")
    
    coords = _coords_from_sequence(match.group())
    return FrontDoor(coords)


def _rooms_from_sequence(seq: str) -> List[Room]:
    regex = r"<Room \d+>(<Coord \d+>)+"
    result = []

    for match in re.finditer(regex, seq):
        room_type_match = re.match(r"<Room (\d+)>", match.group())
        assert(room_type_match)

        room_type = int(room_type_match.group(1))

        boundary = _coords_from_sequence(match.group())

        result.append(Room(RoomType(room_type), boundary))

    if not result:
        raise ValueError("No rooms in sequence")

    return result


def from_sequence(seq: str, name: str) -> FloorPlan:
    boundary = _boundary_from_sequence(seq)
    doors = _door_from_sequence(seq)
    rooms = _rooms_from_sequence(seq)

    return FloorPlan(name, boundary, doors, rooms)
