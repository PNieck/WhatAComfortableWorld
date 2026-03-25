"""Convert floor plan sequences to FloorPlan objects.

This module provides functions to parse structured sequences containing floor plan
data (boundaries, doors, and rooms) and convert them into FloorPlan objects.
"""

import re
from typing import List

import numpy as np

from .parsing_errors import *
from src.floor_plan import FloorPlan, FrontDoor, Room, RoomType


def _coords_from_sequence(seq: str):
    """
    Extracts and reshapes coordinates from a sequence string.
    
    Parses coordinates in the format <Coord N> from the sequence and returns them
    as a 2D array where each row is an (x, y) coordinate pair.
    
    :param seq: Sequence string containing <Coord N> tokens
    :return: NumPy array of shape (num_coords, 2) containing coordinate pairs
    :raises OddNumberOfCoordinatesError: If the number of coordinates is odd
    """
    matches = re.findall(r"<Coord (\d+)>", seq)

    if len(matches) % 2 != 0:
        raise OddNumberOfCoordinatesError(f"Coordinates number ({len(matches)}) is not divisible by 2")

    coords = np.array(list(map(int, matches)), dtype=np.uint8)
    result = coords.reshape(-1, 2)
    
    return result


def boundary_from_sequence(seq: str) -> np.ndarray:
    """
    Extracts the floor plan boundary from a sequence string.
    
    Parses the boundary starting from <Bound> token and extracts its
    coordinates as a 2D array.
    
    :param seq: Sequence string containing boundary data
    :return: NumPy array of shape (num_vertices, 2) representing the boundary polygon
    :raises NoBoundaryError: If no boundary is found in the sequence
    :raises TooSmallNumberOfCoordinatesError: If boundary has fewer than 3 vertices
    """
    match = re.search(r"<Bound>(<Coord \d+>)+", seq)
    if not match:
        raise NoBoundaryError()
    
    result = _coords_from_sequence(match.group())
    if result.shape[0] < 3:
        raise TooSmallNumberOfCoordinatesError(result.shape[0])
    
    return result


def door_from_sequence(seq: str) -> FrontDoor:
    """
    Extracts the front door from a sequence string.
    
    Parses the door starting from <Door> token and creates a FrontDoor object
    from its coordinate endpoints.
    
    :param seq: Sequence string containing door data
    :return: FrontDoor object representing the front door
    :raises NoFrontDoorsError: If no door is found in the sequence
    :raises CoordinatesNumberForFrontDoorError: If the door does not have exactly 2 endpoints
    """
    match = re.search(r"<Door>(<Coord \d+>)+", seq)
    if not match:
        raise NoFrontDoorsError()
    
    coords = _coords_from_sequence(match.group())
    if coords.shape != (2, 2):
        raise CoordinatesNumberForFrontDoorError(f"Got {coords.size} coordinates, expected is 4")

    return FrontDoor(coords)


def _rooms_from_sequence(seq: str) -> List[Room]:
    """
    Extracts all rooms from a sequence string.
    
    Parses rooms in the format <Room type_id>(<Coord N>)+ where type_id indicates
    the room type, and creates Room objects for each parsed room.
    
    :param seq: Sequence string containing room data
    :return: List of Room objects with their types and boundaries
    :raises TooSmallNumberOfCoordinatesError: If any room has fewer than 3 vertices
    :raises NoRoomsError: If no rooms are found in the sequence
    """
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
    """
    Converts a floor plan sequence string into a FloorPlan object.
    
    Parses a structured sequence string containing boundary, door, and room data
    and constructs a complete FloorPlan object.
    
    :param seq: Sequence string containing floor plan data in the format:
                <Bound>(<Coord N>)+<Door>(<Coord N>)+<Room ID>(<Coord N>)+...
    :param name: Optional name for the floor plan (default: "")
    :return: FloorPlan object with boundary, doors, and rooms
    :raises NoBoundaryError: If boundary is missing from the sequence
    :raises NoFrontDoorsError: If door is missing from the sequence
    :raises NoRoomsError: If no rooms are found in the sequence
    :raises Other parsing errors: Various coordinate and format validation errors
    """
    boundary = boundary_from_sequence(seq)
    doors = door_from_sequence(seq)
    rooms = _rooms_from_sequence(seq)

    return FloorPlan(name, boundary, doors, rooms)