class FloorPlanSequenceParsingError(Exception):
    def __init__(self, message):
        super().__init__(message)


class CoordinatesNumberError(FloorPlanSequenceParsingError):
    def __init__(self, message):
        super().__init__(message)


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
