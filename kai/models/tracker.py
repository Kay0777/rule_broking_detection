from scipy.spatial import distance as dist
from numpy import array as NpArray
from typing import Union

from .classes import Point
from .traffic_light import TrafficLight
from .base import CarID, TrackerBase, TrackerMetaClass


class Tracker(TrackerBase, metaclass=TrackerMetaClass):
    def __init__(self, name: str, maxDisappeared: int, maxDistance: int):
        super().__init__(name=name)

        self.traffic_light: Union[None, TrafficLight] = None
        self.__maxDisappeared = maxDisappeared
        self.__maxDistance = maxDistance

    def Update_Traffic_Light_Info(self, coors: tuple[Point, Point], color: str) -> None:
        if self.traffic_light is None:
            self.traffic_light = TrafficLight(name=self.name)

        # Update Crossroad Traffic Light Color
        self.traffic_light.update_color(color=color)

        # Update Crossroad Traffic Light Coordinate
        self.traffic_light.update_coordinates(coors=coors)

    def __register(self, carCoors: tuple[Point, Point], plateCoors: tuple[Point, Point], disappeared: int) -> None:
        self.counter += 1
        self.cars.add(
            CarID(
                carID=self.counter,
                coors=carCoors,
                plateCoors=plateCoors,
                disappeared=disappeared,
            )
        )

    def __get(self, carID: int) -> CarID:
        # Get Given Car By CarID
        return tuple(car for car in self.cars if car.id == carID)[0]

    def __update(self,
                 carID: int,
                 carNewCoors: Union[tuple[Point, Point], None] = None,
                 carNewVpnCoor: tuple = (),
                 disappeared: int = 0) -> None:
        # By in [1, 2, 3]
        # 1 => Update Registrated Car Coordinate To New Coordinate
        # 2 => Update Registrated Car Plate Number To New Plate Number
        # 3 => Update Registrated Car Disappeared Amount To New Disappeared Amount

        car = self.__get(carID=carID)
        car.plateCoors = carNewVpnCoor

        if carNewCoors is not None:
            car.coors = carNewCoors

        if not disappeared:
            car.disappeared += disappeared

    def __deregister(self, carID: int) -> None:
        # Deregister the object but do not remove the ID from usedIDs
        self.cars.remove(CarID(carID=carID))

    def __detect_plate(self, carCoor: tuple[Point, Point], plates: list[tuple[Point, Point]]) -> tuple:
        for i, coor in enumerate(plates):
            carP1, carP2 = carCoor
            vpnP1, vpnP2 = coor

            checkP1 = carP1.x < vpnP1.x < carP2.x and carP1.y < vpnP1.y < carP2.y
            checkP2 = carP1.x < vpnP2.x < carP2.x and carP1.y < vpnP2.y < carP2.y
            if checkP1 and checkP2:
                del plates[i]
                return coor
        return ()

    def Update_Cars(self, detectedCoorsOfCars: list[tuple[Point, Point]], detectedCoorsOfCarPlates: list[tuple[Point, Point]]) -> None:
        if not self.cars:
            for carCoor in detectedCoorsOfCars:
                vpnCoor = self.__detect_plate(carCoor=carCoor, plates=detectedCoorsOfCarPlates)
                self.__register(carCoors=carCoor, plateCoors=vpnCoor, disappeared=0)
            return None

        cars = [car for car in self.cars]
        carIDs = [car.id for car in cars]

        inputCentroids = NpArray([((p1 + p2) // 2).as_array() for p1, p2 in detectedCoorsOfCars])
        objectCentroids = NpArray([car.center.as_array() for car in cars])  # type: ignore

        D = dist.cdist(objectCentroids, inputCentroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows, usedCols = set(), set()
        for row, col in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            # If Distance is more then Between detected car and old position of car will excepted
            if D[row, col] > self.__maxDistance:
                continue

            carID = carIDs[row]
            carNewCoor = detectedCoorsOfCars[col]
            carNewVpnCoor = self.__detect_plate(carCoor=carNewCoor, plates=detectedCoorsOfCarPlates)

            self.__update(
                carID=carID,
                carNewCoors=carNewCoor,
                carNewVpnCoor=carNewVpnCoor,
                disappeared=0)

            usedRows.add(row)
            usedCols.add(col)

        if D.shape[0] < D.shape[1]:
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for col in unusedCols:
                newCarCoor = detectedCoorsOfCars[col]
                newCarVpnCoor = self.__detect_plate(carCoor=carNewCoor, plates=detectedCoorsOfCarPlates)

                self.__register(
                    carCoors=newCarCoor,
                    plateCoors=newCarVpnCoor,
                    disappeared=0)
        else:
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            for row in unusedRows:
                carID = carIDs[row]
                car = self.__get(carID=carID)
                car.disappeared += 1

                if car.disappeared > self.__maxDisappeared:
                    # Removing Car From Tracker
                    self.__deregister(carID=carID)
        return None


if __name__ == '__main__':
    import cv2
    from typing import OrderedDict
    import numpy as np

    coors1 = [
        [406.0, 949.0, 914.0, 1520.0], [1198.0, 532.0, 1524.0, 839.0], [637.0, 496.0, 975.0, 811.0], [1214.0, 239.0, 1461.0, 453.0],
        [797.0, 238.0, 1005.0, 417.0], [895.0, 33.0, 1031.0, 158.0], [1585.0, 115.0, 1755.0, 256.0], [1193.0, 19.0, 1322.0, 130.0]]

    file = 'frames/10.42.7.72/44.npy'
    data1 = np.load(file)
    for coor in coors1:
        x1, y1, x2, y2 = [int(i) for i in coor]
        cv2.rectangle(data1, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cars = OrderedDict()
    counter = 1
    for coor in coors1:
        cars[counter] = CarID(counter, coor)  # type: ignore
        counter += 1

    coors2 = [
        [1201.0, 509.0, 1514.0, 803.0], [440.0, 899.0, 924.0, 1516.0], [807.0, 225.0, 1009.0, 399.0], [653.0, 474.0, 976.0, 774.0],
        [1214.0, 226.0, 1453.0, 431.0], [899.0, 27.0, 1032.0, 146.0], [1611.0, 110.0, 1767.0, 241.0], [1193.0, 12.0, 1318.0, 118.0]]

    file = 'frames/10.42.7.72/45.npy'
    data2 = np.load(file)
    for coor in coors2:
        x1, y1, x2, y2 = [int(i) for i in coor]
        cv2.rectangle(data2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv2.imshow('car1', data1)
    cv2.imshow('car2', data2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # inputCentroids = np.zeros((len(coors2), 2), dtype="int")
    # for (i, (startX, startY, endX, endY)) in enumerate(coors2):
    #     cX = int((startX + endX) / 2.0)
    #     cY = int((startY + endY) / 2.0)
    #     inputCentroids[i] = (cX, cY)

    # objectIDs = list(cars.keys())
    # objectCentroids = [car.center for car in cars.values()]

    # D = dist.cdist(np.array(objectCentroids), inputCentroids) # type: ignore
    # rows = D.min(axis=1).argsort()
    # cols = D.argmin(axis=1)[rows]
    # print('_____________________________________')

    # usedRows = set()
    # usedCols = set()

    # for (row, col) in zip(rows, cols):
    #     if row in usedRows or col in usedCols:
    #         continue

    #     if D[row, col] > MAX_DISTANCE:  # Check if distance exceeds threshold
    #         continue

    #     objectID = objectIDs[row]
    #     cars[objectID] = CarID(1, coors=coors2[col])

    #     # self.__bboxes[objectID] = detectedCoorsOfCars[col]
    #     # self.__disappeared[objectID] = 0
    #     # usedRows.add(row)
    #     # usedCols.add(col)
    # _______________________________________________________________________________

    # unusedRows = set(range(0, D.shape[0])).difference(usedRows)
    # unusedCols = set(range(0, D.shape[1])).difference(usedCols)
    # if D.shape[0] >= D.shape[1]:
    #     for row in unusedRows:
    #         objectID = objectIDs[row]
    #         self.__disappeared[objectID] += 1
    #         if self.__disappeared[objectID] > self.__maxDisappeared:
    #             self.__deregister(objectID)
    # else:
    #     for col in unusedCols:
    #         self.register(inputCentroids[col], detectedCoorsOfCars[col])

    # ct1 = Tracker(name='ct11')
    # ct2 = Tracker(name='ct12')
    # ct3 = Tracker(name='ct12')

    # print(id(ct1))
    # print(id(ct2))
    # print(id(ct3))
