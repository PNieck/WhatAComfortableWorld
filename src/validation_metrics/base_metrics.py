from typing import List

from .parsability_rate import ParsabilityRate
from .geometry_validity import GeomValidityRate
from .coverage import CoverageTest
from .rooms_overlapping import RoomsOverlappingTest


class BaseMetrics:
    def __init__(self):
        self.parsability = ParsabilityRate()
        self.validity = GeomValidityRate()
        self.coverage = CoverageTest()
        self.overlapping = RoomsOverlappingTest()

        self.examples_cnt = 0
        self.correct_examples = 0

    def measure(self, batch: List[str]):
        self.examples_cnt += len(batch)

        floor_plans = self.parsability.parse(batch)
        floor_plans = self.validity.filter_out_invalid(floor_plans)
        floor_plans = self.coverage.filter_out(floor_plans)
        floor_plans = self.overlapping.filter_out(floor_plans)

        self.correct_examples += len(floor_plans)

    def filter_out(self, batch: List[str]):
        floor_plans = self.parsability.parse(batch)
        floor_plans = self.validity.filter_out_invalid(floor_plans)
        floor_plans = self.coverage.filter_out(floor_plans)
        floor_plans = self.overlapping.filter_out(floor_plans)

        return floor_plans

    def success_rate(self):
        return self.correct_examples / self.examples_cnt

    def print_results(self):
        print(f"Base metrics: {self.correct_examples}/{self.examples_cnt} {self.success_rate()}%")

