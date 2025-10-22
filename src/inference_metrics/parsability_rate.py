from typing import Dict, List

from src.floor_plan import FloorPlan
from src.sequence import from_sequence
from src.sequence.parsing_errors import FloorPlanSequenceParsingError


class ParsabilityRate:
    def __init__(self):
        self.examples_cnt = 0
        self.invalid_seq = 0
        self.error_types: Dict[str, int] = {}

    def parse(self, sequences: List[str]) -> List[FloorPlan]:
        result = []
        
        for seq in sequences:
            self.examples_cnt += 1
            
            try:
                floor_plan = from_sequence(seq)
            except FloorPlanSequenceParsingError as err:
                self.invalid_seq += 1
                self._register_error(err)
                continue

            result.append(floor_plan)
        
        return result
    
    def _register_error(self, err: FloorPlanSequenceParsingError):
        error_name = err.__class__.__name__

        if error_name in self.error_types:
            self.error_types[error_name] += 1
        else:
            self.error_types[error_name] = 1
    
    def rate(self) -> float:
        return (self.examples_cnt - self.invalid_seq) / self.examples_cnt
