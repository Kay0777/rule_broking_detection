from .base import (
    ViolationDetectorBase,
    ViolationDetectorMetaClass
)

from typing import Union

from kai.models.base import CarID
from kai.models import Point

COUNT_OF_MISSING_FRAMES: int = 20

COUNT_OF_INCORRECT_DIRECTION: int = 5
INCORRECT_DIRECTION_DIFF: float = 1.2

COUNT_OF_SHRINK_TOUCHING_TO_1_1_LINE: float = 0.15
COUNT_OF_SHRINK_LINE_IN_BOXES: float = 0.25


class ViolationDetector(ViolationDetectorBase, metaclass=ViolationDetectorMetaClass):
    def __init__(self, name: str, configs: dict) -> None:
        super().__init__(name=name, configs=configs)

        self.laFQueue: int = 0
        self.trackedCars: dict[int, dict] = {}
        self.trackedCarDirects: dict = {}
        self.stopLineCrossingFrames: dict = {}
        self.inDirects: dict = {}

        self.trackedCarsStatuses: dict = {}

        self.carsLastTwoCoors: dict = {}
        self.detectViolations: dict = {}

        self.violations: dict[str, set] = self.create_violations()

    # Done
    def __updating_cars(self, laFQueue: int, car: CarID) -> None:
        if car.id not in self.trackedCars.keys():
            carMetaData: dict = {
                'first_seen': laFQueue,
                'last_seen': laFQueue,
                'frame_indices': [laFQueue],
                'bboxes': [car.coors],
                'pbboxes': [car.plateCoors],
                'missing_frames': 0,
                'violation_frame': []
            }
            self.trackedCars[car.id] = carMetaData
        else:
            carMetaData: dict = self.trackedCars[car.id]
            carMetaData.update({
                'last_seen': laFQueue,
                'missing_frames': 0
            })
            carMetaData['frame_indices'].append(laFQueue)
            carMetaData['bboxes'].append(car.coors)
            carMetaData['pbboxes'].append(car.plateCoors)

    # Done
    def Update_Trackers(self, laFQueue: int, cars: set[CarID]) -> list[tuple]:
        for car in cars:
            self.__updating_cars(laFQueue=laFQueue, car=car)

        # Updating Tracked Cars Missing Frames Count
        currentCars: set[int] = set({car.id for car in cars})
        previousCars: set[int] = set(self.trackedCars.keys())
        diffCars: set[int] = previousCars ^ currentCars
        for diffCar in diffCars:
            self.trackedCars[diffCar]['missing_frames'] += 1

        # Removing Losted Cars
        lostCarIDs: set[int] = {
            carID for carID, metaData in self.trackedCars.items()
            if metaData['missing_frames'] > COUNT_OF_MISSING_FRAMES}

        tasks: list[tuple[str, int, dict, set]] = []
        for lostCarID in lostCarIDs:
            hasTheCarBrokenAnyRules = any(lostCarID in self.violations[violation] for violation in self.violations)
            isTheCarPlateDetecedAnyTime = any(self.trackedCars[lostCarID]['pbboxes'])
            if hasTheCarBrokenAnyRules and isTheCarPlateDetecedAnyTime:
                # Need Initialize Task
                tasks.append((
                    self.name,
                    lostCarID,
                    self.trackedCars[lostCarID],
                    self.detectViolations[lostCarID]))

            # Removing Car From Tracking Violation
            del self.trackedCars[lostCarID]
            del self.detectViolations[lostCarID]

            # Removing Car From Direction Violation
            if lostCarID in self.trackedCarDirects:
                del self.trackedCarDirects[lostCarID]

            # Removing Car From Stop Line Violation
            if lostCarID in self.stopLineCrossingFrames:
                del self.stopLineCrossingFrames[lostCarID]

            # Removing Car From In Direct Violation
            if lostCarID in self.inDirects:
                del self.inDirects[lostCarID]
        return tasks

    # Done
    def Check_Violations(self, laFQueue: int, tlColor: str, cars: set[CarID]) -> None:
        for car in cars:
            if car.id not in self.carsLastTwoCoors:
                self.carsLastTwoCoors[car.id] = [None, None]
            # Only update if the new bbox is different from the last one
            if self.carsLastTwoCoors[car.id][-1] != car.coors:
                carLastTwoCoors: list = self.carsLastTwoCoors[car.id]
                carLastTwoCoors[0], carLastTwoCoors[1] = carLastTwoCoors[1], car.coors

            carViolations: set = self.detectViolations.get(car.id, set())
            if 'red_light' not in carViolations:
                tlViolation, violationFrameID = self.__traffick_light_violation(car=car, tlColor=tlColor, laFQueue=laFQueue)
                if tlViolation is not None:
                    self.__recoding_violation(car=car, vtype=tlViolation, laFQueue=violationFrameID)

            if '1.1_line' not in carViolations and self.__1_1_line_violation(car=car):
                self.__recoding_violation(car=car, vtype='1.1_line', violationFrameID=laFQueue)

            if 'incorrect_direction' not in carViolations and self.__incorrect_direction_violation(car=car):
                self.__recoding_violation(car=car, vtype='incorrect_direction', violationFrameID=laFQueue)

    # ____________________________________________________________________
    #       C H E C K I N G   V I O L A T I O N S      #
    # Done
    def __recoding_violation(self, car: CarID, vtype: str, violationFrameID: int) -> None:
        # Record the violation if not already detected
        self.violations[vtype].add(car.id)

        detectCarViolation: set = self.detectViolations.setdefault(car.id, set())
        detectCarViolation.add(vtype)

        trackedCarViolations: list = self.trackedCars[car.id]['violation_frame']
        trackedCarViolations.append((vtype, violationFrameID))

    # Done
    def __traffick_light_violation(self, car: CarID, tlColor: str, laFQueue: int) -> tuple[Union[None, str], Union[None, int]]:
        if car.id not in self.trackedCarsStatuses:
            self.trackedCarsStatuses[car.id] = []

        stopLine: tuple[Point, Point] = self.lines_configs['stop_line'][0]
        redLine: tuple[Point, Point] = self.lines_configs['red_line'][0]

        trackedCarStatus: list[tuple[str, str]] = self.trackedCarsStatuses.get(car.id)
        statuses: set[str] = set(status[0] for status in trackedCarStatus)

        if 'stop' not in statuses and self.__is_line_between_bboxes(car=car, line=stopLine):
            trackedCarStatus.append(('stop', tlColor))
            self.stopLineCrossingFrames[car.id] = laFQueue

        if 'red' not in statuses and self.__is_line_between_bboxes(car=car, line=redLine):
            trackedCarStatus.append(('red', tlColor))

        # Check for 'red_light' violation (exact sequence: [('stop', 'Red'), ('red', 'Red')])
        if trackedCarStatus == [('stop', 'Red'), ('red', 'Red')] or trackedCarStatus == [('stop', 'Red'), ('red', 'RedYellow')]:
            return 'red_light', laFQueue
        elif len(trackedCarStatus) >= 2 and (trackedCarStatus[0] == ('stop', 'Red') or trackedCarStatus[0] == ('stop', 'RedYellow')):
            return 'stop_line', self.stopLineCrossingFrames.get(car.id, laFQueue)
        # Check for 'stop_line' violation (exact sequence: [('stop', 'Red'), ('red', '!Red')])
        return None, None

    # Done
    def __1_1_line_violation(self, car: CarID) -> bool:
        carCoors: tuple[Point, Point] = car.coors

        modifiedX1 = round(carCoors[0].x + COUNT_OF_SHRINK_TOUCHING_TO_1_1_LINE * (carCoors[1].x - carCoors[0].x))
        modifiedX2 = round(carCoors[1].x - COUNT_OF_SHRINK_TOUCHING_TO_1_1_LINE * (carCoors[1].x - carCoors[0].x))

        carEdge = [modifiedX1, carCoors[0].y, modifiedX2, carCoors[1].y]

        for line in self.lines_configs['1.1_line']:
            _line: tuple[Point, Point] = line
            p1, p2 = _line

            modifiedLine = [p1.x, p2.y, p2.x, p2.y]
            if self.__do_lines_intersect(line1=carEdge, line2=modifiedLine):
                return True
        return False

    # Done
    def __incorrect_direction_violation(self, car: CarID) -> bool:
        if car.id not in self.trackedCarDirects:
            # Initialize a new list for this obj_id if it doesn't exist
            self.trackedCarDirects[car.id] = []
        if car.id not in self.inDirects:
            self.inDirects[car.id] = []

        # Getting farthest 1.1 line
        p1, p2 = self.farthest_1_1_Line

        trackedCarDirect: list = self.trackedCarDirects[car.id]
        if car.coors[1].y > min(p1.y, p2.y):
            trackedCarDirect.append(car.coors)
        if len(trackedCarDirect) < COUNT_OF_INCORRECT_DIRECTION:
            return False

        # Extract first and last bounding boxes
        firstCarCoors: tuple[Point, Point] = trackedCarDirect[0]
        lastCarCoors: tuple[Point, Point] = trackedCarDirect[-1]
        if firstCarCoors[0].y < lastCarCoors[0].y and INCORRECT_DIRECTION_DIFF * firstCarCoors[0].y < lastCarCoors[0].y:
            direction = "coming closer"
        elif lastCarCoors[0].y < firstCarCoors[0].y and INCORRECT_DIRECTION_DIFF * lastCarCoors[0].y < firstCarCoors[0].y:
            direction = "moving away"
        else:
            direction = "standing"

        m = (p2.y - p1.y)/(p2.x - p1.x)
        b = p1.y - m * p1.x
        carInDirect: list = self.inDirects[car.id]
        for carDirect in trackedCarDirect:
            x = (carDirect[1].y - b) / m
            if direction == 'coming closer':
                carInDirect.append(carDirect[0].x > x)
            elif direction == 'moving away':
                carInDirect.append(carDirect[1].x < x)
            else:
                carInDirect.append(False)
        return COUNT_OF_INCORRECT_DIRECTION < sum(carInDirect)
    # ____________________________________________________________________

    # ____________________________________________________________________
    #    H E L P E R S   T O   C H E C K I N G   V I O L A T I O N S     #
    @property
    def farthest_1_1_Line(self) -> list[Point, Point]:
        return self.lines_configs['1.1_line'][0]

    # Done
    def __is_line_between_bboxes(self, car: CarID, line: tuple[Point, Point]) -> bool:
        carLastTwoCoors = self.carsLastTwoCoors[car.id]
        if carLastTwoCoors[0] is None:
            return False

        carlastCoor: tuple[Point, Point] = carLastTwoCoors[0]

        topY = round(car.coors[1].y - COUNT_OF_SHRINK_LINE_IN_BOXES * (carlastCoor[1].y - carlastCoor[0].y))
        bottomY = round(carlastCoor[1].y - COUNT_OF_SHRINK_LINE_IN_BOXES * (carlastCoor[1].y - carlastCoor[0].y))

        topY = min(topY, bottomY)
        bottomY = max(topY, bottomY)

        currentTopY, currentBottomY = line[0].y, line[1].y
        return self.__does_line_intersect_or_between(
            currentTopY=currentTopY,
            currentBottomY=currentBottomY,
            topY=topY,
            bottomY=bottomY)

    # Done
    def __does_line_intersect_or_between(self, currentTopY: int, currentBottomY: int, topY: float, bottomY: float) -> bool:
        check1 = currentTopY >= topY and currentTopY <= bottomY
        check2 = currentBottomY >= topY and currentBottomY <= bottomY
        check3 = currentTopY >= topY and currentBottomY <= bottomY
        check4 = currentBottomY >= topY and currentTopY <= bottomY

        if check1 or check2 or check3 or check4:
            return True
        return False

    # Done
    def __do_lines_intersect(self, line1: list[int], line2: list[int]):
        """
        Check if two line segments intersect.
        Each line is defined by four coordinates (x1, y1, x2, y2).
        """
        # Unpack points
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Calculate determinants
        det1 = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        det2 = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        det3 = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)

        # Check if lines are parallel
        if det1 == 0:
            return False  # Lines are parallel

        # Calculate intersection point
        t = det2 / det1
        u = -det3 / det1

        # Check if intersection is within line segments
        return 0 <= t <= 1 and 0 <= u <= 1
    # ____________________________________________________________________
