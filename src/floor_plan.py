"""Core data structures representing floor plans and their components.

Provides classes for representing floor plan geometries including rooms, boundaries,
and front doors.
"""

from enum import Enum
from typing import List

import numpy as np

from shapely import Polygon, LineString, LinearRing


_SCALE_FACTOR = 18/256


class RoomType(Enum):
    """Enumeration of room types in floor plans.
    
    Defines all possible room types with integer identifiers for encoding
    in sequences and models.
    """
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
    """
    Represents a room in a floor plan.
    
    Stores room geometry (boundary coordinates), type, and provides methods
    for computing area and converting between coordinate systems.
    """

    @property
    def corners_cnt(self):
        return self.boundary.shape[0]
    
    @property
    def area(self) -> float:
        """Returns the area of the room polygon in square meters.
        
        :return: Room area in scaled units
        """
        return self.polygon().area

    def __init__(self, type: RoomType, boundary):
        """
        Initialize a room.
        
        :param type: RoomType enum value
        :param boundary: NumPy array of shape (n, 2) with corner coordinates
        """
        self.type = type
        self.boundary = boundary

    def polygon(self) -> Polygon:
        """
        Returns shapely polygon representing room boundary scaled to meters
        
        :return: Shapely polygon
        :rtype: Polygon
        """

        return Polygon(self.boundary * _SCALE_FACTOR)
    
    def normalize(self):
        """
        Makes room vertices counterclockwise
        
        :param self: Description
        """
        ring = LinearRing(self.boundary)
        if not ring.is_ccw:
            self.boundary = self.boundary[::-1]

        # Check regularization
        ring = LinearRing(self.boundary)
        assert ring.is_ccw
    
    @classmethod
    def from_polygon(cls, polygon: Polygon, type: RoomType) -> 'Room':
        """
        Create a Room from a Shapely polygon.
        
        Converts polygon coordinates from scaled units back to integer coordinates,
        
        :param polygon: Shapely Polygon
        :param type: RoomType for the room
        :return: New Room instance
        """
        scaled_polygon = np.array(polygon.exterior.coords) / _SCALE_FACTOR
        scaled_polygon = scaled_polygon[:-1]                  # Remove duplicated last point
        scaled_polygon = scaled_polygon.astype(np.uint8)

        return cls(type, scaled_polygon)


class FrontDoor:
    """
    Represents the front door of a floor plan.
    """
    @property
    def x1(self):
        """X coordinate of first door endpoint."""
        return self.corners[0, 0]
    
    @property
    def y1(self):
        """Y coordinate of first door endpoint."""
        return self.corners[0, 1]
    
    @property
    def x2(self):
        """X coordinate of second door endpoint."""
        return self.corners[1, 0]
    
    @property
    def y2(self):
        """Y coordinate of second door endpoint."""
        return self.corners[1, 1]
    
    @property
    def xs(self):
        """Array of X coordinates for both endpoints."""
        return self.corners[:, 0]
    
    @property
    def ys(self):
        """Array of Y coordinates for both endpoints."""
        return self.corners[:, 1]
    
    @property
    def corner1(self):
        """First endpoint coordinates as array."""
        return self.corners[0]
    
    @property
    def corner2(self):
        """Second endpoint coordinates as array."""
        return self.corners[1]

    def __init__(self, corners: np.ndarray):
        """
        Initialize a front door.
        
        :param corners: NumPy array of shape (2, 2) with [x, y] coordinates for each endpoint
        """
        assert corners.shape == (2, 2), "Invalid array dimensions"
        self.corners = corners
    
    def polygon(self):
        """
        Return the door as a Shapely LineString scaled to meters.
        
        :return: Shapely LineString representing the door
        """
        return LineString(self.corners * _SCALE_FACTOR)

    @classmethod
    def from_xy(cls, x1: int, y1: int, x2: int, y2: int):
        """
        Create a FrontDoor from individual coordinates.
        
        :param x1: X coordinate of first endpoint
        :param y1: Y coordinate of first endpoint
        :param x2: X coordinate of second endpoint
        :param y2: Y coordinate of second endpoint
        :return: New FrontDoor instance
        """
        return cls(np.array([[x1, y1], [x2, y2]], np.int32))


class FloorPlan:
    """
    Represents a complete floor plan.
    """

    MAX_COORDINATE = 255

    @property
    def corners_cnt(self) -> int:
        """Returns the number of corners in the floor plan boundary.
        
        :return: Number of boundary corners (vertices)
        """
        return self.boundary.shape[0]
    
    @property
    def room_cnt(self) -> int:
        """Returns the number of rooms in the floor plan.
        
        :return: Number of rooms
        """
        return len(self.rooms)

    def __init__(self, name: str, boundary, front_door: FrontDoor, rooms: List[Room]):
        """
        Initialize a floor plan.
        
        :param name: Identifier for the floor plan
        :param boundary: NumPy array of shape (n, 2) with floor plan corner coordinates
        :param front_door: FrontDoor object
        :param rooms: List of Room objects
        """
        self.name: str = name
        self.boundary: np.ndarray = boundary
        self.front_door: FrontDoor = front_door
        self.rooms: List[Room] = rooms

    def polygon(self) -> Polygon:
        """
        Returns shapely polygon representing floor plan boundary scaled to meters
        
        :return: Shapely polygon
        :rtype: Polygon
        """
        return Polygon(self.boundary * _SCALE_FACTOR)
    
    def area(self) -> float:
        """
        Return the total area of the floor plan in square meters.
        
        :return: Floor plan area in scaled units
        """
        return self.polygon().area
    
    def rooms_polygons(self) -> List[Polygon]:
        """
        Return all room boundaries as Shapely polygons.
        
        :return: List of Shapely Polygons, one per room
        """
        return [room.polygon() for room in self.rooms]
    
    def rooms_of_type(self, room_type: RoomType) -> List[Room]:
        """
        Find all rooms of a specific type.
        
        :param room_type: RoomType to filter by
        :return: List of Room objects matching the type
        """
        return [room for room in self.rooms if room.type == room_type]
    
    def rooms_of_types(self, room_types: set[RoomType]) -> List[Room]:
        """
        Find all rooms whose type is in a set of room types.
        
        :param room_types: Set of RoomType values to filter by
        :return: List of Room objects with types in the set
        """
        return [room for room in self.rooms if room.type in room_types]
    
    def normalize(self):
        """
        1. Makes boundary vertices counterclockwise
        2. Orients door the same way as walls
        3. Ensures that door are cover by the wall between first and last vertex
        
        :param self: Description
        """
        
        ring = LinearRing(self.boundary)
        if not ring.is_ccw:
            self.boundary = self.boundary[::-1]

        # Check if boundary is ccw
        ring = LinearRing(self.boundary)
        assert ring.is_ccw

        self._orient_door_to_wall()
        self._rotate_corners_to_doors()

        for room in self.rooms:
            room.normalize()


    def _boundary_walls(self):
        """
        Generate LineStrings for each edge of the boundary.
        
        :return: List of Shapely LineString objects, one per boundary edge
        """
        return [
            LineString([self.boundary[i], self.boundary[i+1]])
            for i in range(-1, self.boundary.shape[0]-1)
        ]


    def _nearest_wall_to_door(self):
        """
        Find the boundary wall that contains (covers) the front door.
        
        :return: NumPy array of the wall's endpoints
        :raises Exception: If the door is not covered by any wall
        """
        walls = self._boundary_walls()
        door_polygon = LineString(self.front_door.corners)

        for wall in walls:
            if wall.covers(door_polygon):
                return np.array(wall.coords, np.int32)
            
        raise Exception("Front door are not covered by any wall")


    def _orient_door_to_wall(self):
        """
        Ensure the door direction aligns with its wall direction.
        
        Reverses the door endpoints if necessary so the door vector points
        in the same direction as the wall vector.
        """
        closes_wall = self._nearest_wall_to_door()

        # Direction vectors
        v_wall = closes_wall[1] - closes_wall[0]
        v_door = self.front_door.corner2 - self.front_door.corner1

        if np.dot(v_wall, v_door) < 0:
            self.front_door = FrontDoor(self.front_door.corners[::-1])

        v_door = self.front_door.corner2 - self.front_door.corner1
        assert np.dot(v_wall, v_door) > 0

    def _rotate_corners_to_doors(self):
        """
        Rolls boundary corners so the door is cover by wall between first and last vertex.
        """
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
        """
        Create a FloorPlan from a Shapely polygon.
        
        Converts polygon coordinates from meters back to integer coordinates,
        removes the duplicated last point, and creates a FloorPlan instance.
        
        :param name: Identifier for the floor plan
        :param boundary: Shapely Polygon of the boundary
        :param front_door: FrontDoor object
        :param rooms: List of Room objects
        :return: New FloorPlan instance
        """
        boundary_array = np.array(boundary.exterior.coords) / _SCALE_FACTOR
        boundary_array = boundary_array[:-1]                  # Remove duplicated last point

        boundary_array = boundary_array.astype(np.int32)

        return cls(name, boundary_array, front_door, rooms)
