"""Conversion utilities for MAT file floor plan format.

Provides functions for converting floor plan data from MATLAB (.mat) file format
to FloorPlan objects.
"""

import numpy as np

from typing import List

from src.floor_plan import FloorPlan, FrontDoor, Room, RoomType


def _boundary_len(floor_plan):
    """Calculate the boundary length accounting for front door corners.
    
    :param floor_plan: Floor plan object with boundary data
    :return: Number of boundary points
    """
    boundary_len = floor_plan.boundary.shape[0] - 2

    # Checking if front door corners belongs to boundary
    if floor_plan.boundary[0, 3] == 0:
        boundary_len += 1

    if floor_plan.boundary[1, 3] == 0:
        boundary_len += 1

    return boundary_len


def _boundary_from_mat(floor_plan):
    """Extract boundary coordinates from MAT file floor plan format.
    
    :param floor_plan: Floor plan object with boundary data
    :return: NumPy array of boundary coordinates (N x 2)
    """
    len = _boundary_len(floor_plan)
    
    result = np.empty((len, 2), dtype=np.int32)

    start = 2
    end = floor_plan.boundary.shape[0]

    x = floor_plan.boundary[start:end, 0]
    y = floor_plan.boundary[start:end, 1]

    end = x.shape[0]
    len = end
    result[0:len] = np.stack((x, y), axis=1)

    if floor_plan.boundary[0, 3] == 0:
        x = floor_plan.boundary[0, 0]
        y = floor_plan.boundary[0, 1]

        result[end] = [x, y]
        end += 1

    if floor_plan.boundary[1, 3] == 0:
        x = floor_plan.boundary[1, 0]
        y = floor_plan.boundary[1, 1]

        result[end] = [x, y]
        end += 1

    return result


def _rooms_from_mat(floor_plan) -> List[Room]:
    """Extract room objects from MAT file floor plan format.
    
    :param floor_plan: Floor plan object with room boundary and type data
    :return: List of Room objects
    """
    room_cnt = len(floor_plan.rBoundary)
    result = [None] * room_cnt

    for i, room in enumerate(floor_plan.rBoundary):
        if room.dtype == np.int32:
            x = room[:, 0]
            y = room[:, 1]
        else:
            x = room[:, 0].astype(np.int32)
            y = room[:, 1].astype(np.int32)

        room_boundary = np.stack((x, y), axis=1)
        room_type = RoomType(floor_plan.rType[i])

        result[i] = Room(room_type, room_boundary)

    return result


def from_mat_file(floor_plan) -> FloorPlan:
    """Convert a floor plan from MAT file format to FloorPlan object.
    
    Extracts boundary, front door, and rooms from the MAT file structure
    and constructs a complete FloorPlan object.
    
    :param floor_plan: Floor plan object from MAT file with boundary, door, and room data
    :return: Constructed FloorPlan object
    """
    boundary = _boundary_from_mat(floor_plan)
    
    door = FrontDoor.from_xy(
        floor_plan.boundary[0, 0],
        floor_plan.boundary[0, 1],
        floor_plan.boundary[1, 0],
        floor_plan.boundary[1, 1]
    )

    rooms = _rooms_from_mat(floor_plan)

    return FloorPlan(
        floor_plan.name,
        boundary,
        door,
        rooms
    )