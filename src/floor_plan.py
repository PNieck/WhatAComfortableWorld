from enum import Enum
from typing import List

import numpy as np

from shapely import Polygon


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
    
    @property
    def area(self) -> float:
        return Polygon(self.boundary).area

    def __init__(self, type: RoomType, boundary):
        self.type = type
        self.boundary = boundary

    def boundary_polygon(self) -> Polygon:
        return Polygon(self.boundary)


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
    def xs(self):
        return self.corners[:, 0]
    
    @property
    def ys(self):
        return self.corners[:, 1]
    
    @property
    def corner1(self):
        return self.corners[0]
    
    @property
    def corner2(self):
        return self.corners[1]

    def __init__(self, corners: np.ndarray):
        assert corners.shape == (2, 2), "Invalid array dimensions"
        self.corners = corners        

    @classmethod
    def from_xy(cls, x1: int, y1: int, x2: int, y2: int):
        return cls(np.array([[x1, y1], [x2, y2]], np.uint8))


class FloorPlan:
    MAX_COORDINATE = 255

    @property
    def boundary_len(self):
        return self.boundary.shape[0]
    
    @property
    def rooms_cnt(self):
        return len(self.rooms)

    def __init__(self, name: str, boundary, front_door: FrontDoor, rooms: List[Room]):
        self.name: str = name
        self.boundary: np.ndarray = boundary
        self.front_door: FrontDoor = front_door
        self.rooms: List[Room] = rooms

    def polygon(self) -> Polygon:
        return Polygon(self.boundary)
    
    def rooms_polygons(self) -> List[Polygon]:
        return [room.boundary_polygon() for room in self.rooms]
