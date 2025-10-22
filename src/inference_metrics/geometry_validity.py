from typing import List

from src.floor_plan import FloorPlan


class GeomValidityRate:
    def __init__(self):
        self.examples_cnt = 0
        self.valid_examples = 0
    
    def filter_out_invalid(self, examples: List[FloorPlan]) -> List[FloorPlan]:
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
        return self.valid_examples / self.examples_cnt

