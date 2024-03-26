from typing import Union
from .classes import Point


class CarID:
    # We known that coors will be absolutly different
    # Creating a CarID Class to classificate cars
    def __init__(
            self,
            carID: int,
            coors: Union[None, tuple[Point, Point]] = None,
            plateCoors: Union[None, tuple[Point, Point]] = None,
            disappeared: int = 0
    ) -> None:
        self.id: int = carID
        self.coors: tuple[Point, Point] = coors
        self.plateCoors: Union[None, tuple[Point, Point]] = plateCoors
        self.disappeared: int = disappeared

    @property
    def center(self) -> Union[Point, None]:
        if self.coors is not None:
            return (self.coors[0] + self.coors[1]) // 2
        return None

    def __eq__(self, __value: object) -> bool:
        return self.id == __value.id  # type: ignore

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"[Car: {self.id}]"

    def __str__(self) -> str:
        return f"[Car: {self.id}]"

    def __del__(self) -> None:
        del self.id
        del self.coors
        del self.plateCoors
        del self.disappeared


class TrackerMetaClass(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        name: str = kwargs['name'] if 'name' in kwargs else args[0]
        if name not in cls._instances:
            cls._instances[name] = super().__call__(*args, **kwargs)
        return cls._instances[name]


class TrackerBase:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.counter: int = 0

        self.cars: set[CarID] = set()

    def __repr__(self) -> str:
        return f'[{self.__class__.__name__}: {self.name}]'

    def __str__(self) -> str:
        return f'[{self.__class__.__name__}: {self.name}]'
