from typing import List
import itertools

import tokens
from src.floor_plan import FloorPlan, FrontDoor, Room


def _boundary_sequence(plan: FloorPlan) -> List[str]:
    result = [None] * (plan.corners_cnt * 2 + 1)
    
    result[0] = tokens.BOUNDARY_TOKEN

    for i, corner in enumerate(plan.boundary):
        result[2*i+1] = tokens.coord_token(corner[0])
        result[2*i+2] = tokens.coord_token(corner[1])

    return result


def _door_sequence(door: FrontDoor) -> List[str]:
    return [
        tokens.DOOR_TOKEN,
        tokens.coord_token(door.x1),
        tokens.coord_token(door.y1),
        tokens.coord_token(door.x2),
        tokens.coord_token(door.y2),
    ]


def _room_sequence(room: Room) -> List[str]:
    result = [None] * (room.corners_cnt * 2 + 1)

    result[0] = tokens.room_token(room.type.value)

    for i, corner in enumerate(room.boundary):
        result[2*i + 1] = tokens.coord_token(corner[0])
        result[2*i + 2] = tokens.coord_token(corner[1])

    return result


def to_sequence(plan: FloorPlan) -> List[str]:
    boundary_seq = _boundary_sequence(plan)
    door_seq = _door_sequence(plan.front_door)
    rooms_seqs = [_room_sequence(room) for room in plan.rooms]

    room_seq = itertools.chain.from_iterable(rooms_seqs)

    return list(itertools.chain(boundary_seq, door_seq, room_seq))