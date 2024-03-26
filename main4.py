from multiprocessing import Process, Lock, Value
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
from utils1 import camera_ips
import time
from os import path as OsPath

import os


def detection_each_frame(cameraIPs) -> None:
    def world(ip: str) -> None:
        print('On process with thread in world func:', os.getpid())

        time.sleep(3)
        print(f'Frame detected: || {ip} ||')

    print('On process with thread in main func:', os.getpid())

    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.map(world, cameraIPs)


def analyze_each_active_frame():
    pass


def connect_to_camera(camera, counter) -> None:
    # print('Camera... {}'.format(cameraIP.ip))
    # cap = cv2.VideoCapture(cameraIP.url)
    # if not cap.isOpened():
    #     print(f"Error: Could not open camera {cameraIP}.")
    #     return
    time.sleep(0.6)
    print('Camera connected succesfully... {}'.format(camera.ip), counter)

    while True:
        print(f"in Process: {id(counter)}, {os.getpid()}, {counter.value}")
        time.sleep(0.8)
        counter.value += 1
        print()
    # ret, frame = cap.read()

    # if not ret:
    #     print(f"Error: Could not read frame from camera {cameraIP}.")
    #     break

    # image_path = OsPath.join(cameraIP.folder, f'{counter}.jpg')
    # # print(image_path)
    # saving_each_frame_with_turbojpeg(
    #     image_path=image_path,
    #     data=frame
    # )
    # counter += 1

    # # Store the active frame in the shared dictionary
    # preprocessed_image = preprocess_image(frame)
    # input_tensor = image_to_tensor(preprocessed_image)

    # active_frames[cameraIP.ip] = {
    #     "count": counter,
    #     "data": frame,
    #     "input_tensor": input_tensor,
    # }
    # cap.release()


# def connect_to_camera(args):
#     camera, counter = args
#     print(f'Connecting to camera {camera.ip}...')
#     # Add necessary initialization code here

#     while counter.value < 10:  # Termination condition to exit after a certain number of iterations
#         print(f"Process ID: {os.getpid()}, Counter: {counter.value}")
#         time.sleep(0.8)
#         with counter.get_lock():  # Synchronize access to counter.value
#             counter.value += 1
#     print(f'Camera {camera.ip} disconnected.')


def main():
    ips = camera_ips()

    # threading.excepthook = handle_processes_and_threads_exceptions
    # multiprocessing.excepthook = handle_processes_and_threads_exceptions

    shared_counters: tuple[Value] = (Value('I', 1) for _ in range(len(ips)))

    # with ThreadPoolExecutor() as connectionPools, \
    #         ProcessPoolExecutor() as detectionPools, \
    #         ProcessPoolExecutor() as analyzingPools:
    #     connectionPools.map(connect_to_camera, zip(ips, shared_counters))
    #     # detectionPools.map(detection_each_frame, (ips, ))
    #     # analyzingPools.map(analyze_each_active_frame, ips)

    from concurrent.futures import ProcessPoolExecutor

    def world(a, b):
        print(a, b)

    with ProcessPoolExecutor(max_workers=2) as pool:
        for cameraIP in [1, 2, 3]:
            pool.submit(world, cameraIP, 1)

    # with ThreadPoolExecutor(max_workers=2) as connectionPools:
    #     connectionPools.map(connect_to_camera, *(ips, shared_counters))


if __name__ == '__main__':
    main()
