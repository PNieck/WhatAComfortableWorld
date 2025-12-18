from typing import List

from src.floor_plan import FloorPlan

import shapely

import math


class RoomsOverlappingTest:
    def __init__(self):
        self.overlapping_rate_sum = 0

        self.examples_cnt = 0
        self.correct_floor_plans = 0

    def measure(self, floor_plans: List[FloorPlan]):
        for floor_plan in floor_plans:
            rooms_polygons = floor_plan.rooms_polygons()

            self.examples_cnt += 1
            overlapping_area_sum = 0

            for i in range(len(rooms_polygons)):
                for j in range(i + 1, len(rooms_polygons)):
                    inter = shapely.intersection(rooms_polygons[i], rooms_polygons[j])

                    if not inter.is_empty:
                        overlapping_area_sum += inter.area
                        break
            
            if overlapping_area_sum > 0:
                self.overlapping_rate_sum += overlapping_area_sum / floor_plan.area()

            else:
                self.correct_floor_plans += 1

    def filter_out(self, floor_plans: List[FloorPlan]) -> List[FloorPlan]:
        result = []

        for floor_plan in floor_plans:
            if self.measure_single_floor_plan(floor_plan):
                result.append(floor_plan)
        
        return result
    
    def measure_single_floor_plan(self, floor_plan: FloorPlan) -> bool:
        rooms_polygons = floor_plan.rooms_polygons()

        self.examples_cnt += 1
        overlapping_area_sum = 0

        for i in range(len(rooms_polygons)):
            for j in range(i + 1, len(rooms_polygons)):
                inter = shapely.intersection(rooms_polygons[i], rooms_polygons[j])

                if not inter.is_empty:
                    overlapping_area_sum += inter.area
                    break
        
        if overlapping_area_sum > 0:
            self.overlapping_rate_sum += overlapping_area_sum / floor_plan.area()

        else:
            self.correct_floor_plans += 1
            return True
        
        return False


    def avg_overlapping_rate(self) -> float:
        if self.examples_cnt == 0:
            return math.nan

        return self.overlapping_rate_sum / self.examples_cnt
    
    def correctness_rate(self) -> float:
        if self.examples_cnt == 0:
            return math.nan

        return self.correct_floor_plans / self.examples_cnt
    

    def add_to_metrics(self, metrics: dict):
        metrics["Rooms avg overlapping"] = self.avg_overlapping_rate()
        metrics["Rooms overlapping correctness"] = self.correctness_rate()