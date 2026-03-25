"""Validation metrics for detecting narrow spaces in floor plans.

This module provides functionality to validate that floor plans do not contain
narrow passages that fall below usability standards.
"""

from typing import List
import math

from src.floor_plan import FloorPlan, RoomType
from src.geom_utils import TurnType, turn_type

import shapely


class NarrowSpacesTest:
    """
    Validates that floor plans do not contain narrow passages.
    
    Detects narrow spaces formed by consecutive room corners that are below a minimum
    width threshold. Uses geometric analysis of corner sequences to identify problematic
    narrow passages.
    """

    @property
    def correct_cnt(self):
        """
        Returns the count of floor plans without narrow spaces.
        
        :return: Number of floor plans without problematic narrow spaces
        """
        return self.examples_cnt - self.incorrect_cnt


    def __init__(self, min_width: float = 0.9, neighbors_min_length: float = 0.6):
        """
        Initializes the NarrowSpacesTest validator.
        
        Configures thresholds for detecting narrow passages and sets up counters
        for tracking measurement results.
        
        :param min_width: Minimum acceptable width for passages in meters (default: 0.9)
        :param neighbors_min_length: Minimum length of neighboring segments to flag as narrow in meters (default: 0.6)
        """
        self.min_width = min_width
        self.neighbors_min_length = neighbors_min_length

        self.examples_cnt = 0
        self.incorrect_cnt = 0


    def measure(self, floor_plans: List[FloorPlan]):
        """
        Measures narrow spaces in a batch of floor plans.
        
        Analyzes room boundaries to detect narrow passages formed by consecutive corners.
        A narrow passage is identified when a segment is below min_width and is not on
        the boundary, with neighboring segments both turning the same direction and
        exceeding minimum length.

        Before measuring floor plan has to be simplified by GeometrySimplicityTest
        
        :param floor_plans: List of FloorPlan objects to measure
        """
        self.examples_cnt += len(floor_plans)

        for plan in floor_plans:
            if self.has_narrow_spaces(plan):
                self.incorrect_cnt += 1


    def has_narrow_spaces(self, plan: FloorPlan) -> bool:
        plan_boundary = plan.polygon().exterior

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

                turn1 = turn_type(p0, p1, p2)
                turn2 = turn_type(p1, p2, p3)

                assert turn1 != TurnType.Straight and turn2 != TurnType.Straight, "Unexpected straight turn"

                if turn1 != turn2:
                    continue

                first_segment = shapely.LineString([p0, p1])
                last_segment = shapely.LineString([p2, p3])

                if min(first_segment.length, last_segment.length) <= self.neighbors_min_length:
                    continue

                first_boundary_part = self._boundary_part(first_segment, middle_segment, plan_boundary)
                last_boundary_part = self._boundary_part(last_segment, middle_segment, plan_boundary)

                if (
                    first_boundary_part.length > self.neighbors_min_length and
                    first_segment.covered_by(plan_boundary) and
                    last_boundary_part.length > self.neighbors_min_length
                ):
                    continue

                return True
                    

        return False
    

    def _boundary_part(self, segment, middle_segment, boundary):
        inter = segment.intersection(boundary)

        match inter.geom_type:
            case "LineString" | "Point":
                return inter
            
            case "MultiLineString" | "GeometryCollection" | "MultiPoint":
                for part in inter.geoms:
                    if part.intersects(middle_segment):
                        return part
            
            case _:
                raise Exception(f"Unexpected geometry type {inter.geom_type}")



    def correctness_rate(self) -> float:
        """
        Calculates the rate of floor plans without narrow spaces.
        
        Returns the ratio of floor plans without problematic narrow passages to the
        total number of examples measured.
        
        :return: Correctness rate (0 to 1), or NaN if no examples measured
        """
        if self.examples_cnt == 0:
            return math.nan
        
        return self.correct_cnt / self.examples_cnt


    def add_to_metrics(self, metrics: dict):
        """
        Adds narrow spaces validation metric to a metrics dictionary.
        
        :param metrics: Dictionary to add metrics to (modified in-place)
        """
        metrics["With narrow space rate"] = self.correctness_rate()