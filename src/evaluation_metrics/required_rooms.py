"""Validation of required room types in floor plans.

This module checks generated floor plans for the presence of essential room types
such as kitchens, bathrooms, and sleeping rooms. It accumulates counts of missing
room types and computes correctness metrics.
"""

from typing import List

from src.floor_plan import FloorPlan, RoomType, Room

import math


class RequiredRoomsTest:
    """
    Ensures floor plans contain required room categories.

    Tracks counts of missing kitchens, bathrooms, and sleeping rooms across a
    set of floor plans and computes an overall correctness rate.
    """

    def __init__(self):
        """
        Initializes counters for required room checks.
        """

        self.missing_kitchens_count = 0
        self.missing_bathrooms_count = 0
        self.missing_sleeping_rooms_count = 0

        self.examples_cnt = 0
        self.correct_floor_plans = 0


    def measure(self, floor_plans: List[FloorPlan]):
        """
        Measure required room presence for a batch of floor plans.

        Updates internal counters for each floor plan in the provided list.

        :param floor_plans: List of `FloorPlan` objects to evaluate
        """
        for floor_plan in floor_plans:
            self.examples_cnt += 1

            has_kitchen = self._has_kitchen(floor_plan)
            has_bathroom = self._has_bathroom(floor_plan)
            has_sleeping_room = self._has_sleeping_room(floor_plan)

            if not has_kitchen:
                self.missing_kitchens_count += 1

            if not has_bathroom:
                self.missing_bathrooms_count += 1

            if not has_sleeping_room:
                self.missing_sleeping_rooms_count += 1

            if has_kitchen and has_bathroom and has_sleeping_room:
                self.correct_floor_plans += 1


    def _has_kitchen(self, floor_plan: FloorPlan) -> bool:
        """
        Check whether the floor plan contains a kitchen.

        :param floor_plan: `FloorPlan` to inspect
        :return: True if a kitchen is present, False otherwise
        """
        return any(room.type == RoomType.Kitchen for room in floor_plan.rooms)


    def _has_bathroom(self, floor_plan: FloorPlan) -> bool:
        """
        Check whether the floor plan contains a bathroom.

        :param floor_plan: `FloorPlan` to inspect
        :return: True if a bathroom is present, False otherwise
        """
        return any(room.type == RoomType.Bathroom for room in floor_plan.rooms)


    def _has_sleeping_room(self, floor_plan: FloorPlan) -> bool:
        """
        Check whether the floor plan contains any sleeping room.

        Sleeping rooms include master, second, or living rooms (considered suitable
        sleeping spaces for this validation).

        :param floor_plan: `FloorPlan` to inspect
        :return: True if a sleeping room is present, False otherwise
        """
        return any(self._is_sleeping_room(room) for room in floor_plan.rooms)


    def _is_sleeping_room(self, room: Room) -> bool:
        """
        Determine if a room counts as a sleeping room.

        :param room: `Room` object
        :return: True if the room is a sleeping room type
        """
        return room.type in {RoomType.MasterRoom, RoomType.SecondRoom, RoomType.LivingRoom}


    def correctness_rate(self) -> float:
        """
        Compute the correctness rate for required rooms.

        Returns the fraction of floor plans that contain all required room types.
        If no examples have been measured, returns NaN.

        :return: Correctness rate as a float in [0,1] or NaN if no examples
        """
        if self.examples_cnt == 0:
            return math.nan

        return self.correct_floor_plans / self.examples_cnt


    def add_to_metrics(self, metrics: dict):
        """
        Add required room metrics to a metrics dictionary.

        Populates metrics with overall correctness and counts of missing room types.

        :param metrics: Dictionary to add metrics to (modified in-place)
        """
        metrics["Required rooms"] = self.correctness_rate()
        metrics["Missing kitchens"] = self.missing_kitchens_count
        metrics["Missing bathrooms"] = self.missing_bathrooms_count
        metrics["Missing sleeping rooms"] = self.missing_sleeping_rooms_count


    def print_missing_rooms(self):
        """
        Print counts of missing room types to stdout if any are non-zero.
        """

        if self.missing_kitchens_count > 0:
            print(f"Missing kitchens: {self.missing_kitchens_count}")

        if self.missing_bathrooms_count > 0:
            print(f"Missing bathrooms: {self.missing_bathrooms_count}")

        if self.missing_sleeping_rooms_count > 0:
            print(f"Missing sleeping rooms: {self.missing_sleeping_rooms_count}")