from kai.const import RULES
from typing import Any


class ViolationDetectorMetaClass(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(cls, "instance"):
            cls.instance = super().__call__(*args, **kwargs)
        return cls.instance


class ViolationDetectorBase:
    def __init__(self, name: str, configs: dict) -> None:
        self.name: str = name
        self.lines_configs: dict = configs

    def create_violations(self) -> dict[str, set]:
        violations: dict[str, set] = {key: set() for key in RULES.values()}
        violations.update(
            {
                "incorrect_direction": set(),
            }
        )
        return violations

    def load_lines_configs(self, configs: dict) -> None:
        print("#     L I N E S   C O N F I G S   A R E   L O A D E D     #")
        self.lines_configs = configs

    def __repr__(self) -> str:
        return f"[{self.__class__.__name__}: {self.name}]"

    def __str__(self) -> str:
        return f"[{self.__class__.__name__}: {self.name}]"
