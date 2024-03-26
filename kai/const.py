from typing import TypeAlias

MAX_DISAPPEARED: int = 5
MAX_DISTANCE: int = 100

INPUT_TENSOR_SHAPE: tuple[int, int] = (640, 640)

TRAFFIC_LIGHT_COLORS = [
    "Green",
    "Aniqlanmadi",
    "Red",
    "RedGreen",
    "RedYellow",
    "RedYellowGreen",
    "Yellow",
    "YellowGreen",
]

MNS_KWARGS: dict = {
    "conf_thres": 0.25,
    "iou_thres": 0.7,
    "max_det": 300,
    "agnostic": False,
    "nc": 6,
}

RULES_CLASSES: dict = {
    0: 'car',
    1: 'traffic_light',
    2: 'stop_sign',
    3: 'plate',
    4: 'allowing',
    5: 'additional',
}

ColorType: TypeAlias = dict[str, tuple[int, int, int]]
RuleType: TypeAlias = dict[int, str]

COLORS: ColorType = {
    "1.1_line": (255, 0, 0),  # Blue
    "stop_line": (0, 255, 0),  # Green
    "red_line": (0, 0, 255)  # Red
}

RULES: RuleType = {
    49: "1.1_line",  # key => 1
    50: "stop_line",  # key => 2
    51: "red_line"  # key => 3
}
