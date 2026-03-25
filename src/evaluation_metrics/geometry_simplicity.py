"""Geometric simplicity validation and simplification for floor plans.

This module provides functionality to validate and simplify floor plan geometries by
removing redundant and collinear points from boundaries and rooms.
"""

from typing import List
import math

from src.floor_plan import FloorPlan, Room
from src.geom_utils import remove_collinear_points_ring

import shapely
from shapely import remove_repeated_points


class GeometrySimplicityTest:
    """
    Validates and simplifies floor plan geometries.
    
    Removes redundant points (repeated and collinear) from floor plan boundaries
    and room boundaries to ensure geometric simplicity while preserving the actual shape.
    """

    @property
    def correct_examples_cnt(self) -> int:
        """
        Returns the count of geometrically simple floor plans.
        
        :return: Number of floor plans with already simple geometry
        """
        return self.examples_cnt - self.incorrect_examples_cnt


    def __init__(self):
        """
        Initializes the GeometrySimplicityTest validator.
        
        Sets up counters for tracking total examples and those requiring simplification.
        """
        self.examples_cnt = 0
        self.incorrect_examples_cnt = 0


    def simplify(self, floor_plans: List[FloorPlan]):
        """
        Simplifies floor plan geometries in-place.
        
        Removes repeated and collinear points from both the floor plan boundary and all
        room boundaries. Modifies the floor_plans list in-place with simplified geometries.
        Tracks which floor plans required simplification.
        
        :param floor_plans: List of FloorPlan objects to simplify (modified in-place)
        """
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
        """
        Calculates the geometric simplicity rate.
        
        Returns the ratio of floor plans that were already geometrically simple
        (required no simplification) to the total number of examples.
        
        :return: Simplicity rate (0 to 1), or NaN if no examples processed
        """
        if self.examples_cnt == 0:
            return math.nan

        return self.correct_examples_cnt / self.examples_cnt
    

    def add_to_metrics(self, metrics: dict):
        """
        Adds geometric simplicity metric to a metrics dictionary.
        
        :param metrics: Dictionary to add metrics to (modified in-place)
        """
        metrics["geometric simplicity"] = self.rate()
                
