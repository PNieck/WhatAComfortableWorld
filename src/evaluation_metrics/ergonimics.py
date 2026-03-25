"""Ergonomic validation metrics for floor plan usability.

This module provides functionality to measure ergonomic quality of floor plans by
validating room adjacencies, entrance placement, and other usability metrics.
"""

from typing import List
import math

from src.floor_plan import FloorPlan, RoomType

import shapely
import numpy as np


class ErgonomicsTest:
    """
    Validates ergonomic quality of floor plans.
    
    Measures how well rooms are positioned relative to their functional requirements:
    entrances near doors, kitchens near dining rooms, bathrooms near living areas,
    and balconies accessible from main rooms.
    """

    def __init__(self) -> None:
        """
        Initializes the ErgonomicsTest validator.
        
        Sets up counters for tracking total examples, cumulative losses, and
        perfect floor plans.
        """
        self.examples_cnt = 0
        self.loss_sum = 0

        self.perfect_floor_plans = 0

        self.nan_losses = 0

    def measure(self, floor_plans: List[FloorPlan]):
        """
        Measures ergonomic metrics for a batch of floor plans.
        
        Evaluates all floor plans and accumulates loss and correctness metrics.
        Plans with NaN losses (missing required room types) are counted separately.
        
        :param floor_plans: List of FloorPlan objects to measure
        """
        self.examples_cnt += len(floor_plans)

        for floor_plan in floor_plans:
            loss = self.measure_single(floor_plan)

            if math.isnan(loss):
                self.nan_losses += 1
                continue

            self.loss_sum += loss

            if loss == 0:
                self.perfect_floor_plans += 1

    def measure_single(self, floor_plan: FloorPlan) -> float:
        """
        Measures ergonomic loss for a single floor plan.
        
        Calculates loss metrics for entrances, kitchens, bathrooms, and balconies,
        then returns the normalized average loss. Returns NaN if no valid metrics available.
        
        :param floor_plan: FloorPlan object to measure
        :return: Normalized ergonomic loss (0 is perfect), or NaN if no valid metrics
        """
        losses = np.empty(4)
        rooms_cnt = np.empty(4)

        losses[0], rooms_cnt[0] = self._entrance_loss(floor_plan)
        losses[1], rooms_cnt[1] = self._kitchens_loss(floor_plan)
        losses[2], rooms_cnt[2] = self._bathrooms_loss(floor_plan)
        losses[3], rooms_cnt[3] = self._balconies_loss(floor_plan)

        finite_mask = np.isfinite(losses)

        if not np.any(finite_mask):
            return math.nan

        loss = np.sum(losses[finite_mask]) / np.sum(rooms_cnt)
        return loss


    def correctness_rate(self) -> float:
        """
        Calculates the perfect ergonomic quality rate.
        
        Returns the ratio of floor plans with zero ergonomic loss to the total
        number of examples measured.
        
        :return: Perfect quality rate (0 to 1), or NaN if no examples measured
        """
        if self.examples_cnt == 0:
            return math.nan
        
        return self.perfect_floor_plans / self.examples_cnt


    def _entrance_loss(self, floor_plan: FloorPlan) -> tuple[float, int]:
        """
        Calculates loss for entrance room placement relative to front door.
        
        Measures the distance from each entrance room to the front door.
        Returns NaN if no entrance rooms exist.
        
        :param floor_plan: FloorPlan object to analyze
        :return: Tuple of (total_entrance_loss, num_entrances)
        """
        entrances = floor_plan.rooms_of_type(RoomType.Entrance)
        if not entrances:
            return math.nan, 0

        door_polygon = floor_plan.front_door.polygon()

        loss_sum = 0

        for entrance in entrances:
            entrance_polygon = entrance.polygon()

            room_loss = entrance_polygon.distance(door_polygon)
            loss_sum += room_loss

        return loss_sum, len(entrances)


    def _kitchens_loss(self, floor_plan: FloorPlan) -> tuple[float, int]:
        """
        Calculates loss for kitchen placement near required adjacent rooms.
        
        Kitchens should be adjacent to entrances and dining rooms.
        Returns NaN if no kitchens or required neighbors exist.
        
        :param floor_plan: FloorPlan object to analyze
        :return: Tuple of (total_kitchen_loss, num_kitchens)
        """
        neighbors_types = {RoomType.Entrance, RoomType.DiningRoom}

        return self.calculate_loss(floor_plan, RoomType.Kitchen, neighbors_types)
    

    def _bathrooms_loss(self, floor_plan: FloorPlan) -> tuple[float, int]:
        """
        Calculates loss for bathroom placement near required adjacent rooms.
        
        Bathrooms should be accessible from entrances and main living areas.
        Returns NaN if no bathrooms or required neighbors exist.
        
        :param floor_plan: FloorPlan object to analyze
        :return: Tuple of (total_bathroom_loss, num_bathrooms)
        """
        neighbors_types = {RoomType.Entrance, RoomType.LivingRoom, RoomType.MasterRoom, RoomType.SecondRoom}

        return self.calculate_loss(floor_plan, RoomType.Bathroom, neighbors_types)
    

    def _balconies_loss(self, floor_plan: FloorPlan) -> tuple[float, int]:
        """
        Calculates loss for balcony accessibility from main living rooms.
        
        Balconies should be accessible from one of living rooms, studies, kitchens or bedrooms.
        Returns NaN if no balconies or accessible rooms exist.
        
        :param floor_plan: FloorPlan object to analyze
        :return: Tuple of (total_balcony_loss, num_balconies)
        """
        balconies = floor_plan.rooms_of_type(RoomType.Balcony)
        if not balconies:
            return math.nan, 0
        
        possible_neighbors = floor_plan.rooms_of_types({
            RoomType.LivingRoom,
            RoomType.StudyRoom,
            RoomType.Kitchen,
            RoomType.MasterRoom,
            RoomType.SecondRoom
        })

        if not possible_neighbors:
            return math.nan, 0

        possible_neighbors_polygons = [pn.polygon() for pn in possible_neighbors]

        loss_sum = 0

        for balcony in balconies:
            balcony_polygon = balcony.polygon()
            room_loss = self._min_distance(balcony_polygon, possible_neighbors_polygons)

            loss_sum += room_loss

        return loss_sum, len(balconies)


    def calculate_loss(self, floor_plan: FloorPlan, main_type: RoomType, neighbors_types: set[RoomType]) -> tuple[float, int]:
        """
        Calculates loss based on room adjacency requirements.
        
        Measures distance from main room type to nearest required neighbor types.
        Returns NaN if main rooms or neighbors don't exist.
        
        :param floor_plan: FloorPlan object to analyze
        :param main_type: Primary room type to evaluate
        :param neighbors_types: Set of required adjacent room types
        :return: Tuple of (total_loss, num_main_rooms)
        """
        main_rooms = floor_plan.rooms_of_type(main_type)
        if not main_rooms:
            return math.nan, 0

        neighbors = floor_plan.rooms_of_types(neighbors_types)
        if not neighbors:
            return math.nan, 0

        main_rooms_polygons = [room.polygon() for room in main_rooms]
        neighbors_polygons = [room.polygon() for room in neighbors]
        
        main_rooms_centroids = [geom.centroid for geom in main_rooms_polygons]
        neighbors_centroids = [geom.centroid for geom in neighbors_polygons]
        
        tree = shapely.STRtree(main_rooms_centroids)
        nearest = tree.nearest(neighbors_centroids)
        nearest = np.array(nearest)

        loss_sum = 0

        for i, polygon in enumerate(main_rooms_polygons):
            indices = np.where(nearest == i)[0]
            if indices.size == 0:
                continue

            sum = 0
            for index in indices:
                sum += polygon.distance(neighbors_polygons[index])

            room_loss = sum / len(indices)

            loss_sum += room_loss

        return loss_sum, len(main_rooms)
    

    def _min_distance(self, r1: shapely.Polygon, r2: List[shapely.Polygon]) -> float:
        """
        Finds the minimum distance from one polygon to a list of polygons.
        
        :param r1: Polygon to measure from
        :param r2: List of polygons to measure to
        :return: Minimum distance to any polygon in r2
        """
        min_distance = math.inf

        for polygon2 in r2:
            distance = r1.distance(polygon2)
            if distance < min_distance:
                min_distance = distance

                if min_distance == 0:
                    break

        return min_distance
    
    def avg_loss(self) -> float:
        """
        Calculates the average ergonomic loss across all measured floor plans.
        
        Returns the mean of all valid losses. Returns NaN if no examples were measured.
        
        :return: Average ergonomic loss, or NaN if no examples measured
        """
        if self.examples_cnt == 0:
            return math.nan
        
        return self.loss_sum / self.examples_cnt
