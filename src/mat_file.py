import numpy as np

from typing import List

from src.floor_plan import FloorPlan, FrontDoor, Room, RoomType


def _boundary_len(floor_plan):
    boundary_len = floor_plan.boundary.shape[0] - 2

    # Checking if front door corners belongs to boundary
    if floor_plan.boundary[0, 3] == 0:
        boundary_len += 1

    if floor_plan.boundary[1, 3] == 0:
        boundary_len += 1

    return boundary_len


def _boundary_from_mat(floor_plan):
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
    room_cnt = len(floor_plan.rBoundary)
    result = [None] * room_cnt

    for i, room in enumerate(floor_plan.rBoundary):
        if room.size == 0:
            print(floor_plan.name)
            continue

        x = room[:, 0]
        y = room[:, 1]

        room_boundary = np.stack((x, y), axis=1)
        room_type = RoomType(floor_plan.rType[i])

        result[i] = Room(room_type, room_boundary)

    return result


def from_mat_file(floor_plan) -> FloorPlan:
    boundary = _boundary_from_mat(floor_plan)
    
    door = FrontDoor(
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