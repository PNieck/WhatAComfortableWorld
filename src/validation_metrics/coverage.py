from typing import List

from src.floor_plan import FloorPlan

import shapely

import math


class CoverageTest:
    def __init__(self):
        self.coverage_rate_sum = 0
        self.overfilling_rate_sum = 0

        self.examples_cnt = 0
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

            coverage_rate = inter_area / boundary_area
            overfilling_rate = diff_area / boundary_area

            self.coverage_rate_sum += coverage_rate
            self.overfilling_rate_sum += overfilling_rate

            self.examples_cnt += 1

            if boundary_area == inter_area and diff_area == 0:
                self.correct_floor_plans += 1

    def avg_coverage_rate(self):
        if self.examples_cnt == 0:
            return math.nan
        
        return self.coverage_rate_sum / self.examples_cnt

    def avg_overfilling_rate(self) -> float:
        if self.examples_cnt == 0:
            return math.nan

        return self.overfilling_rate_sum / self.examples_cnt
    
    def correctness_rate(self) -> float:
        if self.examples_cnt == 0:
            return math.nan

        return self.correct_floor_plans / self.examples_cnt

    def add_to_metrics(self, metrics: dict):
        metrics["Boundary avg coverage rate"] = self.avg_coverage_rate()
        metrics["Boundary avg overfilling rate"] = self.avg_overfilling_rate()
        metrics["Boundary coverage correctness rate"] = self.correctness_rate()
