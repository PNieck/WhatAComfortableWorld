from enum import Enum
from typing import List

import numpy as np

from shapely import Polygon, LineString, LinearRing


_SCALE_FACTOR = 18/256


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
    def corners_cnt(self):
        return self.boundary.shape[0]
    
    @property
    def area(self) -> float:
        return self.polygon().area

    def __init__(self, type: RoomType, boundary):
        self.type = type
        self.boundary = boundary

    def polygon(self) -> Polygon:
        return Polygon(self.boundary * _SCALE_FACTOR)
    
    def regularize(self):
        if np.all(self.boundary[0] == self.boundary[-1]):
            self.boundary = self.boundary[:-1]
            assert False
        
        ring = LinearRing(self.boundary)
        if not ring.is_ccw:
            self.boundary = self.boundary[::-1]

        ring = LinearRing(self.boundary)
        assert ring.is_ccw
    
    @classmethod
    def from_polygon(cls, polygon: Polygon, type: RoomType) -> 'Room':
        scaled_polygon = np.array(polygon.exterior.coords) / _SCALE_FACTOR
        scaled_polygon = scaled_polygon[:-1]                  # Remove duplicated last point
        scaled_polygon = scaled_polygon.astype(np.uint8)

        return cls(type, scaled_polygon)


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
    
    def polygon(self):
        return LineString(self.corners * _SCALE_FACTOR)

    @classmethod
    def from_xy(cls, x1: int, y1: int, x2: int, y2: int):
        return cls(np.array([[x1, y1], [x2, y2]], np.int32))


class FloorPlan:
    MAX_COORDINATE = 255

    @property
    def corners_cnt(self):
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
        return Polygon(self.boundary * _SCALE_FACTOR)
    
    def area(self) -> float:
        return self.polygon().area
    
    def rooms_polygons(self) -> List[Polygon]:
        return [room.polygon() for room in self.rooms]
    
    def rooms_of_type(self, room_type: RoomType) -> List[Room]:
        return [room for room in self.rooms if room.type == room_type]
    
    def rooms_of_types(self, room_types: set[RoomType]) -> List[Room]:
        return [room for room in self.rooms if room.type in room_types]
    
    def regularize(self):
        if np.all(self.boundary[0] == self.boundary[-1]):
            self.boundary = self.boundary[:-1]
            assert False
        
        ring = LinearRing(self.boundary)
        if not ring.is_ccw:
            self.boundary = self.boundary[::-1]

        ring = LinearRing(self.boundary)
        assert ring.is_ccw

        self._orient_door_to_wall()
        self._rotate_corners_to_doors()

        for room in self.rooms:
            room.regularize()


    def _boundary_walls(self):
        return [
            LineString([self.boundary[i], self.boundary[i+1]])
            for i in range(-1, self.boundary.shape[0]-1)
        ]


    def _closes_wall_to_door(self):
        walls = self._boundary_walls()
        door_polygon = LineString(self.front_door.corners)

        for wall in walls:
            if wall.covers(door_polygon):
                return np.array(wall.coords, np.int32)
            
        raise Exception("Front door are not covered by any wall")


    def _orient_door_to_wall(self):
        closes_wall = self._closes_wall_to_door()

        # Direction vectors
        v_wall = closes_wall[1] - closes_wall[0]
        v_door = self.front_door.corner2 - self.front_door.corner1

        if np.dot(v_wall, v_door) < 0:
            self.front_door = FrontDoor(self.front_door.corners[::-1])

        v_door = self.front_door.corner2 - self.front_door.corner1
        assert np.dot(v_wall, v_door) > 0

    def _rotate_corners_to_doors(self):
        walls = self._boundary_walls()
        door_polygon = LineString(self.front_door.corners)

        i = 0
        for wall in walls:
            if wall.covers(door_polygon):
                break

            i+= 1

        if i > 0:
            self.boundary = np.roll(self.boundary, -i, axis=0)

        walls = self._boundary_walls()

        for wall in walls:
            if wall.covers(door_polygon):
                return
            
            assert False

    
    @classmethod
    def from_polygon(cls, name: str, boundary: Polygon, front_door: FrontDoor, rooms: List[Room]) -> 'FloorPlan':
        boundary_array = np.array(boundary.exterior.coords) / _SCALE_FACTOR
        boundary_array = boundary_array[:-1]                  # Remove duplicated last point

        boundary_array = boundary_array.astype(np.int32)

        return cls(name, boundary_array, front_door, rooms)
