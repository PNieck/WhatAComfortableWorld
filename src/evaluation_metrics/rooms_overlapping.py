"""Validation metrics for detecting overlapping rooms in floor plans.

This module provides functionality to measure and validate whether rooms in generated
floor plans overlap, calculating overlap rates and correctness metrics.
"""

from typing import List

from src.floor_plan import FloorPlan

import shapely

import math


class RoomsOverlappingTest:
    """
    Validates and measures room overlapping in floor plans.
    
    Tracks overlapping areas between rooms and calculates metrics including
    average overlapping rate and correctness rate for floor plan validation.
    """

    def __init__(self):
        """
        Initializes the RoomsOverlappingTest validator.
        
        Sets up internal counters for tracking overlapping areas and correct floor plans.
        """
        self.overlapping_rate_sum = 0

        self.examples_cnt = 0
        self.correct_floor_plans = 0


    def measure(self, floor_plans: List[FloorPlan]):
        """
        Measures overlapping areas in a batch of floor plans.
        
        Calculates the total overlapping area between rooms for each floor plan and
        updates metrics accordingly.
        
        :param floor_plans: List of FloorPlan objects to measure
        """
        for floor_plan in floor_plans:
            rooms_polygons = floor_plan.rooms_polygons()

            self.examples_cnt += 1
            overlapping_area_sum = 0

            for i in range(len(rooms_polygons)):
                for j in range(i + 1, len(rooms_polygons)):
                    inter = shapely.intersection(rooms_polygons[i], rooms_polygons[j])

                    if not inter.is_empty:
                        overlapping_area_sum += inter.area
            
            if overlapping_area_sum > 0:
                self.overlapping_rate_sum += overlapping_area_sum / floor_plan.area()

            else:
                self.correct_floor_plans += 1


    def filter_out_invalid(self, floor_plans: List[FloorPlan]) -> List[FloorPlan]:
        """
        Filters out floor plans with overlapping rooms.
        
        Returns only the floor plans that have no overlapping rooms, excluding those
        with room intersections from the result.
        
        :param floor_plans: List of FloorPlan objects to filter
        :return: List of valid floor plans with no overlapping rooms
        """
        result = []

        for floor_plan in floor_plans:
            if self.has_no_overlaps(floor_plan):
                result.append(floor_plan)
        
        return result


    def has_no_overlaps(self, floor_plan: FloorPlan) -> bool:
        """
        Returns whether the floor plan is valid (has no overlapping rooms).
        
        :param floor_plan: FloorPlan object to measure
        :return: True if floor plan is valid (no overlaps), False otherwise
        """
        rooms_polygons = floor_plan.rooms_polygons()

        for i in range(len(rooms_polygons)):
            for j in range(i + 1, len(rooms_polygons)):
                inter = shapely.intersection(rooms_polygons[i], rooms_polygons[j])

                if inter.area > 0.0:
                    return False
                    
        return True


    def avg_overlapping_rate(self) -> float:
        """
        Calculates the average overlapping rate across all measured floor plans.
        
        Returns the ratio of total overlapping area to total floor plan area,
        averaged across all examples.
        
        :return: Average overlapping rate (0 to 1), or NaN if no examples measured
        """
        if self.examples_cnt == 0:
            return math.nan

        return self.overlapping_rate_sum / self.examples_cnt


    def correctness_rate(self) -> float:
        """
        Calculates the correctness rate for floor plans.
        
        Returns the ratio of valid floor plans (with no overlaps) to total floor plans measured.
        
        :return: Correctness rate (0 to 1), or NaN if no examples measured
        """
        if self.examples_cnt == 0:
            return math.nan

        return self.correct_floor_plans / self.examples_cnt


    def add_to_metrics(self, metrics: dict):
        """
        Adds overlapping validation metrics to a metrics dictionary.
        
        Adds two metrics: the average overlapping rate and the correctness rate
        for room overlapping validation.
        
        :param metrics: Dictionary to add metrics to (modified in-place)
        """
        metrics["Rooms avg overlapping"] = self.avg_overlapping_rate()
        metrics["Rooms overlapping correctness"] = self.correctness_rate()