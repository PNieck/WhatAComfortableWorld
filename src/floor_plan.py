from enum import Enum
from typing import List

import numpy as np


class RoomType(Enum):
    LivingRoom = 0
    MasterRoom = 1
    Kitchen = 2
    Bathroom = 3
    DiningRoom = 4
    ChildRoom = 5
    StudyRoom = 6
    SecondRoom = 7
    GuestRoom = 8
    Balcony = 9
    Entrance = 10
    Storage = 11
    WallIn = 12


class Room:
    @property
    def boundary_len(self):
        return self.boundary.shape[0]

    def __init__(self, type: RoomType, boundary):
        self.type = type
        self.boundary = boundary


class FrontDoor:
    @property
    def x1(self):
        return self.corners[0, 0]
    
    @property
    def y1(self):
        return self.corners[0, 1]
    
    @property
    def x2(self):
        return self.corners[1, 0]
    
    @property
    def y2(self):
        return self.corners[1, 1]
    
    @property
    def corner1(self):
        return self.corners[0]
    
    @property
    def corner2(self):
        return self.corners[1]

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.corners = np.array([[x1, y1], [x2, y2]], np.int32)


class FloorPlan:
    @property
    def boundary_len(self):
        return self.boundary.shape[0]

    def __init__(self, name: str, boundary, front_door: FrontDoor, rooms: List[Room]):
        self.name = name
        self.boundary = boundary
        self.front_door = front_door
        self.rooms = rooms
