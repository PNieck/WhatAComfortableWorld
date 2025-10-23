from typing import List

from src.floor_plan import FloorPlan

import shapely


class CoverageTest:
    def __init__(self):
        self.total_area = 0
        self.total_room_occupied_area = 0
        self.total_area_outside_boundary = 0
        self.correct_floor_plans = 0

    def measure(self, floor_plans: List[FloorPlan]):
        for floor_plan in floor_plans:
            boundary_polygon = floor_plan.polygon()
            rooms_polygons = floor_plan.rooms_polygons()

            rooms_union = shapely.union_all(rooms_polygons)
            intersection = shapely.intersection(boundary_polygon, rooms_union)
            difference = shapely.difference(rooms_union, boundary_polygon)

            boundary_area = boundary_polygon.area
            inter_area = intersection.area
            diff_area = difference.area

            self.total_area += boundary_area
            self.total_room_occupied_area += inter_area
            self.total_area_outside_boundary += diff_area

            if boundary_area == inter_area and diff_area == 0:
                self.correct_floor_plans += 1

    
    def coverage_rate(self) -> float:
        return self.total_room_occupied_area / self.total_area
    
    def area_outside_rate(self) -> float:
        return self.total_area_outside_boundary / self.total_area
