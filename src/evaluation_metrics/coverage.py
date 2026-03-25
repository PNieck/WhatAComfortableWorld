"""Validation metrics for room coverage within floor plan boundaries.

This module provides functionality to measure and validate whether rooms adequately
cover the floor plan boundary and whether they extend beyond it, calculating coverage
and overfilling rates.
"""

from typing import List

from src.floor_plan import FloorPlan

import shapely

import math


class CoverageTest:
    """
    Validates room coverage within floor plan boundaries.
    
    Measures how well rooms cover the boundary area and detects cases where rooms
    extend beyond the boundary. Calculates coverage rate, overfilling rate, and
    overall correctness metrics.
    """

    def __init__(self):
        """
        Initializes the CoverageTest validator.
        
        Sets up internal counters for tracking coverage rates, overfilling rates,
        and correct floor plans.
        """
        self.coverage_rate_sum = 0
        self.overfilling_rate_sum = 0

        self.examples_cnt = 0
        self.correct_floor_plans = 0

    def measure(self, floor_plans: List[FloorPlan]):
        """
        Measures coverage metrics for a batch of floor plans.
        
        Calculates how well rooms cover the boundary and how much they extend beyond it.
        A floor plan is considered correct if rooms exactly fill the boundary with no gaps
        or overfilling.
        
        :param floor_plans: List of FloorPlan objects to measure
        """
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

    def filter_out_invalid(self, floor_plans: List[FloorPlan]) -> List[FloorPlan]:
        """
        Filters out floor plans with invalid room coverage.
        
        Returns only floor plans where rooms exactly cover the boundary with no gaps
        or overfilling.
        
        :param floor_plans: List of FloorPlan objects to filter
        :return: List of valid floor plans with correct coverage
        """
        result = []
        
        for floor_plan in floor_plans:
            if self.is_valid(floor_plan):
                result.append(floor_plan)

        return result

    def is_valid(self, floor_plan: FloorPlan) -> bool:
        """       
        Evaluates whether rooms exactly cover the boundary with no gaps or overfilling.
        Returns whether the floor plan is valid based on coverage criteria.
        
        :param floor_plan: FloorPlan object to measure
        :return: True if floor plan has perfect coverage, False otherwise
        """
        boundary_polygon = floor_plan.polygon()
        rooms_polygons = floor_plan.rooms_polygons()

        rooms_union = shapely.union_all(rooms_polygons)
        intersection = shapely.intersection(boundary_polygon, rooms_union)
        difference = shapely.difference(rooms_union, boundary_polygon)

        boundary_area = boundary_polygon.area
        inter_area = intersection.area
        diff_area = difference.area

        return boundary_area == inter_area and diff_area == 0


    def avg_coverage_rate(self):
        """
        Calculates the average room coverage rate across measured floor plans.
        
        Returns the ratio of room area intersecting with the boundary to the total
        boundary area, averaged across all examples.
        
        :return: Average coverage rate (0 to 1), or NaN if no examples measured
        """
        if self.examples_cnt == 0:
            return math.nan
        
        return self.coverage_rate_sum / self.examples_cnt

    def avg_overfilling_rate(self) -> float:
        """
        Calculates the average room overfilling rate across measured floor plans.
        
        Returns the ratio of room area extending beyond the boundary to the total
        boundary area, averaged across all examples.
        
        :return: Average overfilling rate (0 to 1), or NaN if no examples measured
        """
        if self.examples_cnt == 0:
            return math.nan

        return self.overfilling_rate_sum / self.examples_cnt
    
    def correctness_rate(self) -> float:
        """
        Calculates the correctness rate for floor plan coverage.
        
        Returns the ratio of floor plans with perfect coverage (no gaps or overfilling)
        to the total number of examples measured.
        
        :return: Correctness rate (0 to 1), or NaN if no examples measured
        """
        if self.examples_cnt == 0:
            return math.nan

        return self.correct_floor_plans / self.examples_cnt

    def add_to_metrics(self, metrics: dict):
        """
        Adds coverage validation metrics to a metrics dictionary.
        
        Adds three metrics: average coverage rate, average overfilling rate,
        and the correctness rate for boundary coverage validation.
        
        :param metrics: Dictionary to add metrics to (modified in-place)
        """
        metrics["Boundary avg coverage rate"] = self.avg_coverage_rate()
        metrics["Boundary avg overfilling rate"] = self.avg_overfilling_rate()
        metrics["Boundary coverage correctness rate"] = self.correctness_rate()
