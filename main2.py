from os import path as OsPath
from time import sleep
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from models import CameraIP
import numpy as np
import cv2
import os
import json
# from turbojpeg import TurboJPEG
import time
from multi_detection2 import process_frame, preprocess_image, image_to_tensor, detection
from line import process_streams
from single_detection import processing_frame
from utils1 import camera_ips
import threading
import multiprocessing
from operator import itemgetter
# from violation import ViolationDetector
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


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
    # with locker:
    np.save(image_path, data)
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
        with locker:
            ret, frame = cap.read()

            if not ret:
                print(f"Error: Could not read frame from camera {cameraIP}.")
                break

            image_path = OsPath.join(cameraIP.folder, f'{counter}')
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
                "input_tensor": input_tensor
            }
    cap.release()


def analyze_active_frame(cameraIP: CameraIP, active_frames: dict, lines_info: list, analyzed_frames: list, boxes: dict) -> None:
    last_frame = None

    while True:
        if not active_frames:
            continue

        print(cameraIP.ip, active_frames.keys())

        # start = time.monotonic()
        # _frames: dict = active_frames.get(cameraIP.ip, None)
        # if _frames is None:
        #     continue

        # if last_frame == _frames.get('count'):
        #     continue

        # # print('Fock**', cameraIP.ip)
        # # process_frame(_frames.get('data'), _frames.get('input_tensor'), cameraIP.ip)
        # # processing_frame(_frames.get('data'), _frames.get('input_tensor'), cameraIP.ip)

        # last_frame = _frames.get('count')
        # # frame_ids.append(_frames.get('count'))
        # end = time.monotonic()
        # diff = (end - start) * 1000
        # print ('All time:', diff)
        # print()


def main():
    lines_info = []
    cameraIPs: list[CameraIP] = camera_ips()

    global locker
    locker = threading.Lock()

    for cameraIP in cameraIPs:
        lines_info.append(get_lines_info(cameraIP.url, cameraIP.ip))

    max_worker_count = len(cameraIPs)
    active_frames = {}  # Dictionary for shared data among threads
    analyzed_frames = ()
    boxes = ()

    connect_threads = []
    for cameraIP in cameraIPs:
        thread = threading.Thread(target=connect_to_camera, args=(cameraIP, active_frames))
        thread.start()
        connect_threads.append(thread)

    analyze_process = threading.Thread(target=detection, args=(cameraIPs, active_frames))
    analyze_process.start()


if __name__ == "__main__":
    main()
