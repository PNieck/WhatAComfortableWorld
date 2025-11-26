from typing import List

from src.floor_plan import FloorPlan, RoomType, Room


class RequiredRoomsTest:
    def __init__(self):
        self.missing_kitchens_count = 0
        self.missing_bathrooms_count = 0
        self.missing_sleeping_rooms_count = 0

        self.examples_cnt = 0
        self.correct_floor_plans = 0


    def measure(self, floor_plans: List[FloorPlan]):
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
        return any(room.type == RoomType.Kitchen for room in floor_plan.rooms)


    def _has_bathroom(self, floor_plan: FloorPlan) -> bool:
        return any(room.type == RoomType.Bathroom for room in floor_plan.rooms)


    def _has_sleeping_room(self, floor_plan: FloorPlan) -> bool:
        return any(self._is_sleeping_room(room) for room in floor_plan.rooms)


    def _is_sleeping_room(self, room: Room) -> bool:
        return room.type in {RoomType.MasterRoom, RoomType.SecondRoom, RoomType.LivingRoom}


    def correctness_rate(self) -> float:
        if self.examples_cnt == 0:
            return 0.0

        return self.correct_floor_plans / self.examples_cnt


    def add_to_metrics(self, metrics: dict):
        metrics["Required rooms"] = self.correctness_rate()
        metrics["Missing kitchens"] = self.missing_kitchens_count
        metrics["Missing bathrooms"] = self.missing_bathrooms_count
        metrics["Missing sleeping rooms"] = self.missing_sleeping_rooms_count


    def print_missing_rooms(self):

        if self.missing_kitchens_count > 0:
            print(f"Missing kitchens: {self.missing_kitchens_count}")

        if self.missing_bathrooms_count > 0:
            print(f"Missing bathrooms: {self.missing_bathrooms_count}")

        if self.missing_sleeping_rooms_count > 0:
            print(f"Missing sleeping rooms: {self.missing_sleeping_rooms_count}")