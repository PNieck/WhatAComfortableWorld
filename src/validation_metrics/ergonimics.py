from typing import List
import math

from src.floor_plan import FloorPlan, RoomType

import shapely
import numpy as np


class ErgonomicsTest:
    def __init__(self) -> None:
        self.examples_cnt = 0
        self.loss_sum = 0

        self.perfect_floor_plans = 0

        self.nan_losses = 0

    def measure(self, floor_plans: List[FloorPlan]):
        for floor_plan in floor_plans:
            self.examples_cnt += 1
    
            losses = np.empty(4)
            rooms_cnt = np.empty(4)

            losses[0], rooms_cnt[0] = self._entrance_loss(floor_plan)
            losses[1], rooms_cnt[1] = self._kitchens_loss(floor_plan)
            losses[2], rooms_cnt[2] = self._bathrooms_loss(floor_plan)
            losses[3], rooms_cnt[3] = self._balconies_loss(floor_plan)

            finite_mask = np.isfinite(losses)

            if not np.any(finite_mask):
                self.nan_losses += 1
                continue

            loss = np.sum(losses[finite_mask]) / np.sum(rooms_cnt)
            self.loss_sum += loss

            if loss == 0:
                self.perfect_floor_plans += 1


    def correctness_rate(self) -> float:
        if self.examples_cnt == 0:
            return 0.0
        
        return self.perfect_floor_plans / self.examples_cnt


    def _entrance_loss(self, floor_plan: FloorPlan) -> tuple[float, int]:
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
        neighbors_types = {RoomType.Entrance, RoomType.DiningRoom}

        return self.calculate_loss(floor_plan, RoomType.Kitchen, neighbors_types)
    

    def _bathrooms_loss(self, floor_plan: FloorPlan) -> tuple[float, int]:
        neighbors_types = {RoomType.Entrance, RoomType.LivingRoom, RoomType.MasterRoom, RoomType.SecondRoom}

        return self.calculate_loss(floor_plan, RoomType.Bathroom, neighbors_types)
    

    def _balconies_loss(self, floor_plan: FloorPlan) -> tuple[float, int]:
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
        min_distance = math.inf

        for polygon2 in r2:
            distance = r1.distance(polygon2)
            if distance < min_distance:
                min_distance = distance

                if min_distance == 0:
                    break

        return min_distance
    
    def avg_loss(self) -> float:
        if self.examples_cnt == 0:
            return math.nan
        
        return self.loss_sum / self.examples_cnt