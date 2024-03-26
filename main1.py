from os import path as OsPath
from time import sleep
import multiprocessing
from multiprocessing import Lock
from concurrent.futures import ProcessPoolExecutor
from models import CameraIP
import numpy as np
import cv2
import os
import json
# from turbojpeg import TurboJPEG
import time
from multi_detection1 import process_frame, preprocess_image, image_to_tensor
from line import process_streams
from single_detection import processing_frame
from utils1 import camera_ips
import threading
from operator import itemgetter
# from violation import ViolationDetector
from concurrent.futures import ThreadPoolExecutor

locker = Lock()

violation = {}

analyzed_frames_lock = threading.Lock()


def read_lines_from_file(file_path):
    with open(file_path, 'r') as file:
        lines_info = json.load(file)
    return lines_info

# Function to write lines info to file


def write_lines_to_file(file_path, lines_info):
    with open(file_path, 'w') as file:
        json.dump(lines_info, file)


def get_lines_info(rtsp_url, ip_address):
    line_file_path = f'line_{ip_address}.txt'
    if os.path.exists(line_file_path):
        return read_lines_from_file(line_file_path)
    else:
        lines_info = process_streams([rtsp_url])
        write_lines_to_file(line_file_path, lines_info[0])
        return lines_info[0]


def saving_each_frame_with_turbojpeg(image_path: str, data: np.ndarray) -> bool:
    # engine = TurboJPEG()
    with locker:
        _start = time.monotonic()
        # with open(image_path, "wb") as file:
        #     file.write(engine.encode(data, quality=95))
        #     file.close()
        # return True
        np.save(image_path, data)
        _end = time.monotonic()
        print('Save_time:', 1000 * (_end - _start))
        return True


def connect_to_camera(cameraIP: CameraIP, active_frames: dict) -> None:
    print('Camera... {}'.format(cameraIP.ip))
    cap = cv2.VideoCapture(cameraIP.url)
    if not cap.isOpened():
        print(f"Error: Could not open camera {cameraIP}.")
        return
    print('Camera connected succesfully... {}'.format(cameraIP.ip))

    counter = 0
    while True:
        while locker:
            ret, frame = cap.read()

            if not ret:
                print(f"Error: Could not read frame from camera {cameraIP}.")
                break

            image_path = OsPath.join(cameraIP.folder, f'{counter}.m[y]')
            # print(image_path)
            saving_each_frame_with_turbojpeg(
                image_path=image_path,
                data=frame
            )
            counter += 1

            # Store the active frame in the shared dictionary
            preprocessed_image = preprocess_image(frame)
            input_tensor = image_to_tensor(preprocessed_image)

            active_frames[cameraIP.ip] = {
                "count": counter,
                "data": frame,
                "input_tensor": input_tensor,
            }
        cap.release()


def analyze_active_frame(active_frames: dict, lines_info: list, analyzed_frames: list):
    global violation
    last_frame = None

    while True:
        # with locker:
        if active_frames:
            start = time.monotonic()
            frames = [
                (
                    cameraIP,
                    active_frames[cameraIP]['count'],
                    active_frames[cameraIP]['data'],
                    active_frames[cameraIP]["input_tensor"]
                ) for cameraIP in active_frames.keys()
            ]

            sorted_frames = sorted(frames, key=itemgetter(0))

            camera_ip = [ip[0] for ip in sorted_frames]
            frame_id = [frame_id[1] for frame_id in sorted_frames]

            # if last_frame is None or (last_frame[0] != frame_id[0] or last_frame[1] != frame_id[1]):
            #     if len([ips[0] for ips in sorted_frames]) > 1:
            #         if len([tensor[3] for tensor in sorted_frames]) == 2:

            #             _start = time.monotonic()
            #             with analyzed_frames_lock:
            #                 process_frame([frame[2] for frame in sorted_frames], [tensor[3] for tensor in sorted_frames])

            #             _end = time.monotonic()
            #             print('Processed time:', 1000 * (_end - _start))

            # elif len([ips[0] for ips in sorted_frames]) == 1:
            #     with analyzed_frames_lock:
            #             analyzed_frames = processing_frame([frame[2] for frame in sorted_frames], [tensor[3] for tensor in sorted_frames])
            #     # processed_frame, detected_objects, tl_color = analyzed_frames[0]

            # else:
            #     "No camera was inserted"

            # else:
            #     print ("Already finished this frame")
            #     time.sleep(0.05)

            #     last_frame = frame_id

            end = time.monotonic()
            diff = (end - start) * 1000
            print('All time:', diff)
            print()


def main():
    lines_info = []
    cameraIPs: list[CameraIP] = camera_ips()
    desired_size = (640, 640)

    for cameraIP in cameraIPs:
        lines_info.append(get_lines_info(cameraIP.url, cameraIP.ip))

    active_frames = {}  # Dictionary for shared data among threads
    analyzed_frames = []

    # Create threads for each camera connection
    connect_threads = []
    for cameraIP in cameraIPs:
        thread = threading.Thread(target=connect_to_camera, args=(cameraIP, active_frames))
        thread.start()
        connect_threads.append(thread)

    analysis_thread = threading.Thread(target=analyze_active_frame, args=(active_frames, lines_info, analyzed_frames, locker))
    analysis_thread.start()


if __name__ == "__main__":
    main()
