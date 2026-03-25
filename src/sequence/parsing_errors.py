"""Custom exception classes for floor plan sequence parsing errors.

Defines a hierarchy of exceptions for different parsing failure modes during
floor plan sequence tokenization and validation.
"""


class FloorPlanSequenceParsingError(Exception):
    """Base exception for floor plan sequence parsing errors."""
    def __init__(self, message):
        super().__init__(message)


class OddNumberOfCoordinatesError(FloorPlanSequenceParsingError):
    def __init__(self, message):
        super().__init__(message)


class TooSmallNumberOfCoordinatesError(FloorPlanSequenceParsingError):
    def __init__(self, nb: int):
        super().__init__(f"Coordinates number {nb} is too small")


class NoRoomsError(FloorPlanSequenceParsingError):
    def __init__(self, message=""):
        super().__init__(message)


class NoFrontDoorsError(FloorPlanSequenceParsingError):
    def __init__(self, message=""):
        super().__init__(message)


class CoordinatesNumberForFrontDoorError(FloorPlanSequenceParsingError):
    def __init__(self, message):
        super().__init__(message)


class NoBoundaryError(FloorPlanSequenceParsingError):
    def __init__(self, message=""):
        super().__init__(message)
