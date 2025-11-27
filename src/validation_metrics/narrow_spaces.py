from typing import List

from src.floor_plan import FloorPlan
from src.geom_utils import TurnType, turn_type

import shapely


class NarrowSpacesTest:

    @property
    def correct_cnt(self):
        return self.examples_cnt - self.incorrect_cnt


    def __init__(self, min_width: float = 0.9, neighbors_min_length: float = 0.6):
        self.min_width = min_width
        self.neighbors_min_length = neighbors_min_length

        self.examples_cnt = 0
        self.incorrect_cnt = 0


    def measure(self, floor_plans: List[FloorPlan]):
        for plan in floor_plans:
            plan_boundary = plan.polygon().exterior

            correct = True
            self.examples_cnt += 1

            for room in plan.rooms:
                room_coords = room.polygon().exterior.coords
                room_coords = room_coords[:-1]                  # Remove duplicated last point

                for i in range(len(room_coords)):
                    p0 = room_coords[i]
                    p1 = room_coords[i-1]
                    p2 = room_coords[i-2]
                    p3 = room_coords[i-3]

                    middle_segment = shapely.LineString([p1, p2])

                    if middle_segment.length >= self.min_width:
                        continue

                    if middle_segment.covered_by(plan_boundary):
                        continue

                    turn1 = turn_type(p0, p1, p2)
                    turn2 = turn_type(p1, p2, p3)

                    assert turn1 != TurnType.Straight and turn2 != TurnType.Straight, "Unexpected straight turn"

                    if turn1 == turn2:
                        first_segment = shapely.LineString([p0, p1])
                        last_segment = shapely.LineString([p2, p3])

                        if min(first_segment.length, last_segment.length) > self.neighbors_min_length:
                            correct = False
                            self.incorrect_cnt += 1
                            break

                if not correct:
                    break


    def correctness_rate(self) -> float:
        if self.examples_cnt == 0:
            return 0.0
        
        return self.correct_cnt / self.examples_cnt


    def add_to_metrics(self, metrics: dict):
        metrics["With narrow space rate"] = self.correctness_rate()