"""Convert FloorPlan objects to sequence strings.

This module provides functions to convert FloorPlan objects into structured
sequence strings containing boundary, door, and room data.
"""

from typing import List
import itertools

import src.tokens as tokens
from src.floor_plan import FloorPlan, FrontDoor, Room


def _boundary_sequence(plan: FloorPlan) -> List[str]:
    """
    Converts the floor plan boundary to a token sequence.
    
    Creates a token sequence starting with the boundary token followed by
    alternating X and Y coordinate tokens for each corner of the boundary polygon.
    
    :param plan: FloorPlan object containing the boundary
    :return: List of tokens representing the boundary in sequence format
    """
    result = [None] * (plan.corners_cnt * 2 + 1)
    
    result[0] = tokens.BOUNDARY_TOKEN

    for i, corner in enumerate(plan.boundary):
        result[2*i+1] = tokens.coord_token(corner[0])
        result[2*i+2] = tokens.coord_token(corner[1])

    return result


def _door_sequence(door: FrontDoor) -> List[str]:
    """
    Converts a front door to a token sequence.
    
    Creates a token sequence with the door token followed by the X and Y coordinates
    of the two endpoints of the front door.
    
    :param door: FrontDoor object to convert
    :return: List of tokens representing the door in sequence format
    """
    return [
        tokens.DOOR_TOKEN,
        tokens.coord_token(door.x1),
        tokens.coord_token(door.y1),
        tokens.coord_token(door.x2),
        tokens.coord_token(door.y2),
    ]


def _room_sequence(room: Room) -> List[str]:
    """
    Converts a room to a token sequence.
    
    Creates a token sequence starting with a room token (encoding the room type)
    followed by alternating X and Y coordinate tokens for each corner of the room polygon.
    
    :param room: Room object to convert
    :return: List of tokens representing the room in sequence format
    """
    result = [None] * (room.corners_cnt * 2 + 1)

    result[0] = tokens.room_token(room.type.value)

    for i, corner in enumerate(room.boundary):
        result[2*i + 1] = tokens.coord_token(corner[0])
        result[2*i + 2] = tokens.coord_token(corner[1])

    return result


def to_sequence(plan: FloorPlan) -> List[str]:
    """
    Converts a FloorPlan object to a token sequence.
    
    Combines the boundary, front door, and all rooms into a single token sequence.
    The sequence follows the format: boundary tokens + door tokens + rooms tokens.
    
    :param plan: FloorPlan object to convert
    :return: List of tokens representing the complete floor plan in sequence format
    """
    boundary_seq = _boundary_sequence(plan)
    door_seq = _door_sequence(plan.front_door)
    rooms_seqs = [_room_sequence(room) for room in plan.rooms]

    room_seq = itertools.chain.from_iterable(rooms_seqs)

    return list(itertools.chain(boundary_seq, door_seq, room_seq))