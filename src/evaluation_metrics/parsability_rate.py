"""Parsability metrics for floor plan sequence parsing.

This module measures how many sequence strings can be successfully parsed into
`FloorPlan` objects. It tracks counts of examples, invalid sequences, and the
types of parsing errors encountered.
"""

from typing import Dict, List

from src.floor_plan import FloorPlan
from src.sequence import from_sequence
from src.sequence.parsing_errors import FloorPlanSequenceParsingError

import math


class ParsabilityRate:
    """
    Tracks parsability of floor plan sequence strings.

    Accumulates statistics about how many sequences can be parsed into
    `FloorPlan` objects and records parsing error types.
    """

    def __init__(self):
        """
        Initializes the ParsabilityRate tracker.

        Sets counters for total examples, invalid sequences, and a mapping of
        error type names to occurrence counts.
        """
        self.examples_cnt = 0
        self.invalid_seq = 0
        self.error_types: Dict[str, int] = {}

    def parse(self, sequences: List[str]) -> List[FloorPlan]:
        """
        Parse a list of sequence strings into `FloorPlan` objects.

        :param sequences: List of floor plan sequence strings to parse
        :return: List of successfully parsed `FloorPlan` objects
        """
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
        """
        Record a parsing error occurrence.

        Increments the counter for the specific error class name.

        :param err: The parsing error instance to record
        """
        error_name = err.__class__.__name__

        if error_name in self.error_types:
            self.error_types[error_name] += 1
        else:
            self.error_types[error_name] = 1
    
    def rate(self) -> float:
        """
        Compute the parsability rate.

        Returns the fraction of sequences that were successfully parsed. If no
        examples have been recorded, returns NaN.

        :return: Parsability rate as a float in [0,1], or NaN if no examples
        """
        if self.examples_cnt == 0:
            return math.nan

        return (self.examples_cnt - self.invalid_seq) / self.examples_cnt
    
    def add_to_metrics(self, metrics: dict):
        """
        Add the parsability metric to a metrics dictionary.

        :param metrics: Dictionary to add the parsability metric to (modified in-place)
        """
        metrics["parsability"] = self.rate()
