class FloorPlanSequenceParsingError(Exception):
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
