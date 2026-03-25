"""Geometric validity validation for floor plans.

This module provides functionality to validate that floor plan boundaries and rooms
form valid, non-self-intersecting geometries.
"""

from typing import List

from src.floor_plan import FloorPlan

import math


class GeomValidityRate:
    """
    Validates geometric validity of floor plans.
    
    Checks that floor plan boundaries and all rooms form valid geometries without
    self-intersections or other geometric issues.
    """

    def __init__(self):
        """
        Initializes the GeomValidityRate validator.
        
        Sets up counters for tracking total examples and geometrically valid examples.
        """
        self.examples_cnt = 0
        self.valid_examples = 0
    
    def filter_out_invalid(self, examples: List[FloorPlan]) -> List[FloorPlan]:
        """
        Filters out floor plans with invalid geometries.
        
        Validates that the floor plan boundary and all room boundaries are geometrically
        valid (no self-intersections). Returns only valid floor plans.
        
        :param examples: List of FloorPlan objects to validate
        :return: List of geometrically valid FloorPlan objects
        """
        result = []
        
        for example in examples:
            self.examples_cnt += 1

            if not example.polygon().is_valid:
                continue

            rooms_are_valid = True
            for room in example.rooms_polygons():
                if not room.is_valid:
                    rooms_are_valid = False
                    break

            if rooms_are_valid:
                self.valid_examples += 1
                result.append(example)

        return result
    
    def rate(self) -> float:
        """
        Calculates the geometric validity rate.
        
        Returns the ratio of geometrically valid floor plans to the total number
        of examples validated.
        
        :return: Validity rate (0 to 1), or NaN if no examples validated
        """
        if self.examples_cnt == 0:
            return math.nan

        return self.valid_examples / self.examples_cnt
    
    def add_to_metrics(self, metrics: dict):
        """
        Adds geometric validity metric to a metrics dictionary.
        
        :param metrics: Dictionary to add metrics to (modified in-place)
        """
        metrics["geometric validity"] = self.rate()

