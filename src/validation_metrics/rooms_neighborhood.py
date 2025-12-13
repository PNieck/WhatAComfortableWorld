from typing import List
import math

from src.floor_plan import FloorPlan, RoomType

import shapely
import numpy as np


class RoomsNeighborhoodTest():
    def __init__(self):
        self.loss_sum = 0

        self.examples_cnt = 0
        self.correct_floor_plans = 0
        self.perfect_floor_plans = 0

        self.nan_losses = 0

    def measure(self, floor_plans: List[FloorPlan]):
        for floor_plan in floor_plans:
            self.examples_cnt += 1

            losses = np.empty(4)

            losses[0] = self._entrance_loss(floor_plan)
            losses[1] = self._kitchens_loss(floor_plan)
            losses[2] = self._bathrooms_loss(floor_plan)
            losses[3] = self._balconies_loss(floor_plan)

            finite_mask = np.isfinite(losses)

            if not np.any(finite_mask):
                self.nan_losses += 1
                continue

            loss = losses[finite_mask].mean()
            self.loss_sum += loss

            if loss == 0:
                self.perfect_floor_plans += 1


    def avg_loss(self) -> float:
        if self.examples_cnt == 0:
            return math.nan
        
        return self.loss_sum / self.examples_cnt
    

    def add_to_metrics(self, metrics: dict):
        metrics["Avg room neighborhood loss"] = self.avg_loss()


    def _entrance_loss(self, floor_plan: FloorPlan) -> float:
        entrances = floor_plan.rooms_of_type(RoomType.Entrance)
        if not entrances:
            return math.nan

        door_polygon = floor_plan.front_door.polygon()

        room_losses = np.empty(len(entrances))

        for i, entrance in enumerate(entrances):
            entrance_polygon = entrance.polygon()

            room_losses[i] = entrance_polygon.distance(door_polygon)

        return room_losses.mean()

                
    def _kitchens_loss(self, floor_plan: FloorPlan) -> float:
        kitchens = floor_plan.rooms_of_type(RoomType.Kitchen)
        if not kitchens:
            return math.nan

        entrances = floor_plan.rooms_of_type(RoomType.Entrance)
        dining_rooms = floor_plan.rooms_of_type(RoomType.DiningRoom)

        if (not entrances) and (not dining_rooms):
            return math.nan

        entrances_polygons = [entrance.polygon() for entrance in entrances]
        dining_rooms_polygons = [dining_room.polygon() for dining_room in dining_rooms]

        room_losses = np.empty(len(kitchens))

        for i, kitchen in enumerate(kitchens):
            kitchen_polygon = kitchen.polygon()

            min_dist_entrance = self._min_distance(kitchen_polygon, entrances_polygons)
            min_dist_dining_rooms = self._min_distance(kitchen_polygon, dining_rooms_polygons)

            dists = np.array([min_dist_dining_rooms, min_dist_entrance])

            mask = np.isfinite(dists)

            if not np.any(mask):
                room_losses[i] = 0
            else:
                room_losses[i] = dists[mask].mean()

        return room_losses.mean()

    

    def _bathrooms_loss(self, floor_plan: FloorPlan) -> float:
        bathrooms = floor_plan.rooms_of_type(RoomType.Bathroom)
        if not bathrooms:
            return math.nan

        entrances = floor_plan.rooms_of_type(RoomType.Entrance)
        living_rooms = floor_plan.rooms_of_type(RoomType.LivingRoom)
        bedrooms = floor_plan.rooms_of_types({RoomType.MasterRoom, RoomType.SecondRoom})

        if (not entrances) and (not living_rooms) and (not bedrooms):
            return math.nan 

        entrances_polygons = [entrance.polygon() for entrance in entrances]
        living_rooms_polygons = [living_room.polygon() for living_room in living_rooms]
        bedrooms_polygons = [bedroom.polygon() for bedroom in bedrooms]

        room_losses = np.empty(len(bathrooms))

        for i, bathroom in enumerate(bathrooms):
            bathroom_polygon = bathroom.polygon()

            min_dist_entrance = self._min_distance(bathroom_polygon, entrances_polygons)
            min_dist_living_rooms = self._min_distance(bathroom_polygon, living_rooms_polygons)
            min_dist_bedrooms = self._min_distance(bathroom_polygon, bedrooms_polygons)

            dists = np.array([
                min_dist_entrance,
                min_dist_living_rooms,
                min_dist_bedrooms
            ])

            mask = np.isfinite(dists)

            if not np.any(mask):
                room_losses[i] = 0
            else:
                room_losses[i] = dists[mask].mean()

        return room_losses.mean()
        
        
    def _balconies_loss(self, floor_plan: FloorPlan) -> float:
        balconies = floor_plan.rooms_of_type(RoomType.Balcony)
        if not balconies:
            return math.nan
        
        possible_neighbors = floor_plan.rooms_of_types({
            RoomType.LivingRoom,
            RoomType.StudyRoom,
            RoomType.Kitchen,
            RoomType.MasterRoom,
            RoomType.SecondRoom
        })

        possible_neighbors_polygons = [pn.polygon() for pn in possible_neighbors]

        if not possible_neighbors:
            return math.nan

        room_losses = np.empty(len(balconies))

        for i, balcony in enumerate(balconies):
            balcony_polygon = balcony.polygon()
            room_losses[i] = self._min_distance(balcony_polygon, possible_neighbors_polygons)
            
        finite_mask = np.isfinite(room_losses)
        if not np.any(finite_mask):
            return 0
        
        return room_losses[finite_mask].mean()


    def _min_distance(self, r1: shapely.Polygon, r2: List[shapely.Polygon]) -> float:
        min_distance = math.inf

        for polygon2 in r2:
            distance = r1.distance(polygon2)
            if distance < min_distance:
                min_distance = distance

                if min_distance == 0:
                    break

        return min_distance
