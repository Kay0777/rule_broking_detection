from dataclasses import dataclass
from enum import Enum
from config import CONF

__all__ = (
    "Point",
    "ModelCoreType",
    "Rule",
    "Device",
    "ModelType"
)

# @     D A T A C L A S S                                           @ #
# ____________________________________________________________________#


@dataclass
class Point:
    x: int
    y: int

    def __eq__(self, __value: object) -> bool:
        return self.x == __value.x and self.y == __value.y

    def as_point(self) -> tuple[int, int]:
        return (self.x, self.y)

    def as_array(self) -> list[int]:
        return [self.x, self.y]

    def __add__(self, other):
        return Point(
            x=self.x + other.x,
            y=self.y + other.y)

    def __truediv__(self, other):
        return Point(
            x=self.x / other,
            y=self.y / other)

    def __floordiv__(self, other):
        return Point(
            x=self.x // other,
            y=self.y // other)


@dataclass
class ModelCoreType:
    name: str
    model_type: str

    def __str__(self) -> str:
        return f'{self.name}'
# ____________________________________________________________________#


# @     E N U M S                                                   @ #
# ____________________________________________________________________#
class Rule(Enum):
    CAR = 0
    TRAFFIC_LIGHT = 1
    STOP_SIGN = 2
    PLATE = 3
    ALLOWING = 4
    ADDITIONAL = 5


class Device(Enum):
    CPU = "CPU"
    GPU = "GPU"

    def __str__(self) -> str:
        return f'Model: {self.value}'

    @property
    def type(self) -> str:
        return self.value


class ModelType(Enum):
    ALL = ModelCoreType(
        name='All Objects',
        # model_type=CONF['all_objects'],
        model_type=CONF['all_objects'])
    TRAFFIC_LIGHT = ModelCoreType(
        name='Traffic Light',
        # model_type=CONF['traffic_light'],
        model_type=CONF['traffic_light'])

    def __str__(self) -> str:
        return f'Model: {self.value}'

    @property
    def name(self) -> str:
        return self.value.name

    @property
    def type(self) -> str:
        return self.value.model_type
# ____________________________________________________________________#
