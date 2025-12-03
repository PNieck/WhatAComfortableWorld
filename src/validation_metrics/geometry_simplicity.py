from typing import List

from src.floor_plan import FloorPlan, Room
from src.polygons_utils import remove_collinear_points_ring

import shapely
from shapely import remove_repeated_points


class GeometrySimplicityTest:

    @property
    def correct_examples_cnt(self) -> int:
        return self.examples_cnt - self.incorrect_examples_cnt


    def __init__(self):
        self.examples_cnt = 0
        self.incorrect_examples_cnt = 0


    def simplify(self, floor_plans: List[FloorPlan]):      
        for i, plan in enumerate(floor_plans):
            self.examples_cnt += 1
            correct_example = True

            plan_polygon = plan.polygon()

            simplified_boundary = remove_repeated_points(plan_polygon.boundary)
            simplified_boundary = remove_collinear_points_ring(simplified_boundary)

            polygon_simplified = shapely.Polygon(simplified_boundary)
            
            if not shapely.equals_identical(plan_polygon, polygon_simplified):
                correct_example = False
                plan = FloorPlan.from_polygon(plan.name, polygon_simplified, plan.front_door, plan.rooms)
                floor_plans[i] = plan

            for j, room in enumerate(plan.rooms):
                room_polygon = room.polygon()

                room_simplified_boundary = remove_repeated_points(room_polygon.boundary)
                room_simplified_boundary = remove_collinear_points_ring(room_simplified_boundary)

                room_polygon_simplified = shapely.Polygon(room_simplified_boundary)

                if not shapely.equals_identical(room_polygon, room_polygon_simplified):
                    room = Room.from_polygon(room_polygon_simplified, room.type)
                    plan.rooms[j] = room
                    correct_example = False

            if not correct_example:
                self.incorrect_examples_cnt += 1


    def rate(self) -> float:
        if self.examples_cnt == 0:
            return 0.0

        return self.correct_examples_cnt / self.examples_cnt
    

    def add_to_metrics(self, metrics: dict):
        metrics["geometric simplicity"] = self.rate()
                
