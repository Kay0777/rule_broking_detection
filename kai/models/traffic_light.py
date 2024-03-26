from typing import Union

from .classes import Point

class TrafficLight:
    __instances: dict = {}

    def __new__(cls, *args, **kwargs):
        name: str = kwargs['name'] if 'name' in kwargs else args[0]

        if name not in cls.__instances:
            cls.__instances[name] = super(TrafficLight, cls).__new__(cls)
            cls.__instances[name].name = name
        return cls.__instances[name]

    def __init__(self, name: str) -> None:
        self.name: str = name

        self.coors: Union[None, tuple[Point, Point]] = None
        self.color: Union[None, str] = None
    
    def __repr__(self) -> str:
        return f"[Traffic Light: {self.name}]"

    def __str__(self) -> str:
        return f"[Traffic Light: {self.name}]"
    
    def update_coordinates(self, coors: tuple[Point, Point]) -> None:
        self.coors = coors
    
    def update_color(self, color: str) -> None:
        self.color = color


def main():
    tl1 = TrafficLight('10.42.1.72')
    tl2 = TrafficLight('10.42.1.73')

    print(tl1)

    print(tl2)

if __name__ == "__main__":
    main()