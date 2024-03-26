import cv2
import json

from typing import Union
from .models import Point
from os import path as OsPath, mkdir

from config import CONF
from .const import COLORS, RULES


class Line:
    def __init__(self, camera: str) -> None:
        self.camera = camera
        file_format: str = CONF['line_config_file_format']
        file_folder: str = CONF['config_lines_foldername']

        _path: str = OsPath.join(CONF['path'], file_folder)
        if not OsPath.exists(path=_path):
            mkdir(path=_path)
        self.config_filename = OsPath.join(_path, file_format.format(camera))

        self.__configs: dict[str, list[list[list[int]]]] = {
            key: [] for key in RULES.values()
        }

        self.display_size: tuple[int, int] = (640, 640)
        self.current_line_type: Union[None, str] = None

        self.p1: Point = Point(x=-1, y=-1)
        self.p2: Point = Point(x=-1, y=-1)

        self.isDrawing: bool = False
        self.__isConfigAdded: bool = True

    @property
    def config(self) -> dict:
        return self.__sorting_lines_configs()

    def __save_line_configs(self) -> None:
        if not self.__isConfigAdded:
            return

        with open(file=self.config_filename, mode='w') as f:
            json.dump(obj=self.__configs, fp=f, indent=2)
            f.close()

    def __upload_line_configs(self) -> None:
        with open(file=self.config_filename, mode='r') as f:
            self.__configs = json.loads(s=f.read())
            f.close()

    def __sorting_lines_configs(self):
        __sorted_lines_configs: dict = {
            key: [] for key in RULES.values()
        }
        for key, values in self.__configs.items():
            values = sorted(values, key=lambda val: val[0][0])
            for value in values:
                p1, p2 = value
                data = (p1, p2) if p1[0] < p2[0] else (p2, p1)
                __sorted_lines_configs[key].append(
                    [
                        Point(x=data[0][0], y=data[0][1]),
                        Point(x=data[1][0], y=data[1][1])
                    ]
                )
        return __sorted_lines_configs

    def __add_line_info(self, key: str) -> None:
        scale_x = self.shape[1] / self.display_size[0]
        scale_y = self.shape[0] / self.display_size[1]

        p1x, p1y = int(self.p1.x * scale_x), int(self.p1.y * scale_y)
        p2x, p2y = int(self.p2.x * scale_x), int(self.p2.y * scale_y)

        self.__configs[key].append(
            [
                [p1x, p1y],
                [p2x, p2y]
            ]
        )

    def __draw_line(self, event, x, y, flags, param) -> None:
        # Only start drawing if a label is selected
        if self.current_line_type is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing a line
            self.isDrawing = True
            self.p1 = Point(x=x, y=y)

        elif event == cv2.EVENT_MOUSEMOVE and self.isDrawing:
            # Draw a line on a copy of the image to show real-time drawing
            img_copy = self.img.copy()
            cv2.line(
                img=img_copy,
                pt1=self.p1.as_point(),
                pt2=(x, y),
                color=COLORS[self.current_line_type],
                thickness=2,
                lineType=2)
            cv2.imshow(
                winname=self.camera,
                mat=img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing the line
            self.isDrawing = False
            self.p2 = Point(x=x, y=y)

            cv2.line(
                img=self.img,
                pt1=self.p1.as_point(),
                pt2=self.p2.as_point(),
                color=COLORS[self.current_line_type],
                thickness=2,
                lineType=2)
            cv2.imshow(
                winname=self.camera,
                mat=self.img)

            # Add Line Points To Config
            self.__add_line_info(
                key=self.current_line_type)
            self.__isConfigAdded = True

    def load(self, url: str) -> tuple[bool, tuple[int, ...]]:
        cap = cv2.VideoCapture(url)
        # Checking Connection with Camera
        if not cap.isOpened():
            print('#____________________________________________#')
            print(f'Connection is refused Camera: {self.camera}')
            print('#____________________________________________#')
            return False, (0, 0, 0)

        # Configurate capture
        ret, frame = cap.read()
        if not ret:
            print('#____________________________________________#')
            print(f"Failed to grab the first frame from stream: {self.camera}.")
            print('#____________________________________________#')
            return False, (0, 0, 0)

        if OsPath.exists(path=self.config_filename):
            print('Line Config file exists upload it (y) or draw lines (n)!? [y/n]')
            deafult_answer: str = CONF['load_exists_config_file']
            if deafult_answer == 'y':
                self.__upload_line_configs()
                print(f'Camera: {self.camera} Lines Config is uploaded...')
                return True, frame.shape

        print('#____________________________________________#')
        print(f'  Drawing Config Lines on Camera: {self.camera}')
        print('#____________________________________________#')

        # Resize the image for drawing purposes
        self.shape: tuple[int, ...] = frame.shape
        self.img = cv2.resize(frame, self.display_size)

        cv2.namedWindow(winname=self.camera)
        cv2.imshow(winname=self.camera, mat=self.img)

        title: list[str] = [f'{i + 1} => {value}' for i, value in enumerate(RULES.values())]
        keys: set[int] = set(RULES.keys())
        print('#____________________________________________#')
        print('Please select given labels............')
        print(title)
        print('#____________________________________________#')
        while True:
            cv2.setMouseCallback(window_name=self.camera, on_mouse=self.__draw_line)  # type: ignore
            key: int = cv2.waitKey(delay=1) & 0xFF
            if key == ord('q'):
                break

            elif key in keys:
                self.current_line_type = RULES.get(key)
            elif key != 255:
                print('#____________________________________________#')
                print('Please select given labels............')
                print(title)
                print('#____________________________________________#')

        # Saving Lines Configs To File
        self.__save_line_configs()

        # Closing opened windows and terminating connection with the camera
        cv2.destroyAllWindows()
        cap.release()

        return self.__isConfigAdded, frame.shape
