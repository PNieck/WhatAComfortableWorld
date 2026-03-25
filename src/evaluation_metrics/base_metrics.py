"""Base metrics for comprehensive floor plan validation.

This module aggregates multiple validation checks (parsability, geometry validity,
coverage, and room overlapping) to provide overall usability metrics for generated
floor plans.
"""

from typing import List
import math

from .parsability_rate import ParsabilityRate
from .geometry_validity import GeomValidityRate
from .coverage import CoverageTest
from .rooms_overlapping import RoomsOverlappingTest


class BaseMetrics:
    """
    Comprehensive validator for generated floor plans.
    
    Aggregates multiple validation checks including parsability, geometry validity,
    room coverage, and room overlapping to compute overall floor plan usability metrics.
    """

    def __init__(self):
        """
        Initializes the BaseMetrics validator with all sub-validators.
        
        Creates instances of parsability, geometry validity, coverage, and overlapping
        validators to perform comprehensive floor plan validation.
        """
        self.parsability = ParsabilityRate()
        self.validity = GeomValidityRate()
        self.coverage = CoverageTest()
        self.overlapping = RoomsOverlappingTest()

        self.examples_cnt = 0
        self.correct_examples = 0

    def measure(self, batch: List[str]):
        """
        Measures validation metrics on a batch of floor plan sequences.
        
        Processes all sequences through parsability, validity, coverage, and overlapping
        validators in sequence. Counts how many floor plans pass all validation checks.
        
        :param batch: List of floor plan sequence strings to validate
        """
        self.examples_cnt += len(batch)

        floor_plans = self.parsability.parse(batch)
        floor_plans = self.validity.filter_out_invalid(floor_plans)
        floor_plans = self.coverage.filter_out_invalid(floor_plans)
        floor_plans = self.overlapping.filter_out_invalid(floor_plans)

        self.correct_examples += len(floor_plans)

    def filter_out_invalid(self, batch: List[str]):
        """
        Filters a batch of floor plan sequences, returning only valid ones.
        
        Processes all sequences through parsability, validity, coverage, and overlapping
        validators in sequence, returning only those that pass all checks.
        
        :param batch: List of floor plan sequence strings to filter
        :return: List of valid FloorPlan objects that passed all validators
        """
        floor_plans = self.parsability.parse(batch)
        floor_plans = self.validity.filter_out_invalid(floor_plans)
        floor_plans = self.coverage.filter_out_invalid(floor_plans)
        floor_plans = self.overlapping.filter_out_invalid(floor_plans)

        return floor_plans

    def success_rate(self):
        """
        Calculates the overall success rate of floor plan generation.
        
        Returns the ratio of floor plans that passed all validation checks to the
        total number of examples measured.
        
        :return: Success rate as a decimal (0 to 1), or NaN if no examples measured
        """
        if self.examples_cnt == 0:
            return math.nan
        
        return self.correct_examples / self.examples_cnt

    def print_results(self):
        """
        Prints the base metrics validation results.
        
        Outputs a summary line showing the number of correct examples, total examples,
        and the overall success rate as a percentage.
        """
        print(f"Base metrics: {self.correct_examples}/{self.examples_cnt} {self.success_rate()}%")

