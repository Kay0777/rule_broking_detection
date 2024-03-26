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
import time
from multi_detection import process_frame, preprocess_image, image_to_tensor
from line import process_streams
from single_detection import processing_frame
from utils1 import camera_ips
import threading
from operator import itemgetter
from turbojpeg import TurboJPEG
from violationD import ViolationDetector

locker = Lock()

violation = {}


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

# def saving_each_frame_with_turbojpeg(image_path: str, data: np.ndarray) -> bool:
#     engine = TurboJPEG()

#     with locker:
#         _start = time.monotonic()
#         with open(image_path, "wb") as file:
#             file.write(engine.encode(data, quality=95))
#             file.close()
#         # return True
#         # np.save(image_path, data)
#         _end = time.monotonic()
#         print('Save_time:', 1000 * (_end - _start))
#         return True


def connect_to_camera(cameraIP: CameraIP, active_frames: dict) -> None:
    print('Camera... {}'.format(cameraIP.ip))
    cap = cv2.VideoCapture(cameraIP.url)
    if not cap.isOpened():
        print(f"Error: Could not open camera {cameraIP}.")
        return
    print('Camera connected succesfully... {}'.format(cameraIP.ip))

    counter = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"Error: Could not read frame from camera {cameraIP}.")
            break

        image_path = OsPath.join(cameraIP.folder, f'{counter}.jpg')
        # print(image_path)
        # saving_each_frame_with_turbojpeg(
        #     image_path=image_path,
        #     data=frame
        # )
        cv2.imwrite(image_path, frame)
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


def analyze_active_frame(frames: dict, lines_info: list, last_frame: int, analyzed_frames: list):
    global violation
    sorted_frames = sorted(frames, key=itemgetter(0))

    camera_ip = [ip[0] for ip in sorted_frames]
    frame_id = [frame_id[1] for frame_id in sorted_frames]

    start = time.monotonic()
    if last_frame is None or (last_frame[0] != frame_id[0] or last_frame[1] != frame_id[1]):  # type: ignore
        if len([ips[0] for ips in sorted_frames]) > 1:
            if len([tensor[3] for tensor in sorted_frames]) == 2:

                # np.save('trash/temp.npy', sorted_frames[0][3])
                _start = time.monotonic()
                analyzed_frames[0] = process_frame([frame[2] for frame in sorted_frames], [tensor[3] for tensor in sorted_frames])
                _end = time.monotonic()
                print('Processed time:', 1000 * (_end - _start))

                for i, analyzed_frame in enumerate(analyzed_frames[0]):
                    if i not in violation:
                        violation[i] = ViolationDetector()

                    _, detected_objects, tl_color = analyzed_frame
                    if detected_objects:
                        # violation: ViolationDetector = violation_detectors[camera_ip[i]]
                        violation[i].update_tracker(frame_id[i], detected_objects, camera_ip[i])

                        line_positions = {}
                        for line in lines_info[i]:
                            # For other lines, proceed as before
                            if line[2] not in line_positions:
                                line_positions[line[2]] = []
                            line_positions[line[2]].append((line[0], line[1]))
                        violation[i].check_violations(frame_id[i], tl_color, line_positions, detected_objects)

        elif len([ips[0] for ips in sorted_frames]) == 1:
            analyzed_frames[0] = processing_frame([frame[2] for frame in sorted_frames], [tensor[3] for tensor in sorted_frames])
            # processed_frame, detected_objects, tl_color = analyzed_frames[0]

        else:
            "No camera was inserted"

    else:
        print("Already finished this frame")
        time.sleep(0.05)

    last_frame = frame_id  # type: ignore

    end = time.monotonic()
    diff = (end - start) * 1000
    print('All time:', diff)
    print()

    return sorted_frames


# def create_videos(frames):
#     for _ in range(2):  # For demonstration purposes, create videos for 5 iterations
#         print('Create a video...')
#         sleep(1)


def main():
    lines_info = []
    last_frame = None
    cameraIPs: list[CameraIP] = camera_ips()
    desired_size = (640, 640)
    for cameraIP in cameraIPs:
        lines_info.append(get_lines_info(cameraIP.url, cameraIP.ip))
    with multiprocessing.Manager() as manager:
        active_frames = manager.dict()

        analyzed_frames = [None]
        # print(violation_detectors)
        # return
        with ProcessPoolExecutor(max_workers=6) as pool:
            futures = [
                pool.submit(connect_to_camera, cameraIP, active_frames)  # type: ignore
                for cameraIP in cameraIPs
            ]

            while True:
                info = [
                    (
                        cameraIP,
                        active_frames[cameraIP]['count'],
                        active_frames[cameraIP]['data'],
                        active_frames[cameraIP]["input_tensor"]
                    ) for cameraIP in active_frames.keys()
                ]
                if info:
                    # return True
                    analysis_thread = threading.Thread(target=analyze_active_frame, args=(info, lines_info, last_frame, analyzed_frames))
                    analysis_thread.start()
                    analysis_thread.join()

                    # _start = time.monotonic()
                    # for id, frame in enumerate(analyzed_frames[0]):
                    #     # print(frame[0][0])
                    #     # Create a unique window name for each frame
                    #     window_name = f"Frame {id}"

                    #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    #     cv2.resizeWindow(window_name, desired_size)
                    #     display_frame = cv2.resize(frame[0], desired_size)

                    #     # Show each frame in its own window
                    #     cv2.imshow(window_name, display_frame)

                    # _end = time.monotonic()
                    # print('Imshow time:', 1000 * (_end - _start))
                    # print()
                    # frame = analyzed_frames.get(0)
                    # # Create a unique window name for each frame
                    # window_name = f"Frame {id}"

                    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow(window_name, desired_size)
                    # display_frame = cv2.resize(frame, desired_size)

                    # # Show each frame in its own window
                    # cv2.imshow(window_name, display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        for future in futures:
            print(future.result())


if __name__ == '__main__':
    main()
