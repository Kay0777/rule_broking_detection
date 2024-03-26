from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from multiprocessing.managers import Namespace, ListProxy
from multiprocessing import Manager
from queue import Queue

from utils import timeIt
from PIL import Image
import numpy as np
import time
import cv2

from kai.models import Tracker, TensorModel, ModelType, Point
from kai.const import MAX_DISAPPEARED, MAX_DISTANCE
from kai.camera import Camera
from kai.line import Line
from analysis import (
    Handle_Exception,

    Laf_HWC_To_CHW,
    CHW_Laf_To_TenSor,
    Traffic_Light_Color,
    Classification_Tensors,
    Transform_Traffic_Light_Image,
)
from violation import ViolationDetector
from video_creator import Create_Video, Transformer_New_Task_Schema_To_Old_Task_Schema
# from save import save_violation_video_if_needed

import time


def Video_Creator(tasks: Queue) -> None:
    while True:
        time.sleep(1)
        if tasks.empty():
            continue

        newTask: list[tuple[int, int, dict, dict, list[Point]]] = tasks.get()
        if newTask is None:
            # Terminate Video Creator Task
            break

        oldTask: dict = Transformer_New_Task_Schema_To_Old_Task_Schema(task=newTask)
        # Create_Video(task=newTask)
        # save_violation_video_if_needed(
        #     obj_id=oldTask['obj_id'],
        #     frame_id=oldTask['frame_id'],
        #     ip_address=oldTask['ip_address'],
        #     tracker=oldTask['tracker'],
        #     has_violation=oldTask['has_violation'],
        #     detected_violations=oldTask['detected_violations'],
        #     violation_video_path=oldTask['violation_video_path'],
        #     farthest_line=oldTask['farthest_line'],
        # )
        tasks.task_done()
    print('# ________________________________________________________ #')
    print('#       V I D E O   C R E A T O R   I S   C L O S E D      #')
    print('# ________________________________________________________ #')


def Connect_To_Camera(camera: Camera) -> None:
    print(f'Camera: {camera} is connecting...')
    # Connection to the camera with opencv
    cap: cv2.VideoCapture = cv2.VideoCapture(camera.url)
    # If not connection with camera, all the camera tasks will be killed!
    if not cap.isOpened():
        # Terminate the camera for another waiting threads and processes if the camera is not connected.
        camera.isAlive = False

        # Start the camera for opther waiting threads and processes for closing
        camera.run()
        print(f"Error: Could not open camera {camera}.")
        return

    print(f'Camera: {camera} connected !!!')
    # Start the camera for other waiting threads and processes if the camera is connected.
    camera.run()
    # Start reading frame from the camera
    while True:
        # Read connection status and frame from camera
        ret, frame = cap.read()

        # If connection status is False
        # Kill the camera task loop for other tasks
        if not ret:
            print(f"Error: Could not read frame from camera {camera}.")
            camera.isAlive = False
            break

        # Save each frame of the camera
        camera.save(data=frame)

        # Check camera block status
        # If camera blocked and analyzes active frame, active frame will not update to next frame
        if not camera.isBlocked:
            # Create and Update Camera LAF Shared Memory Data
            camera.update_last_active_frame_data(newLafData=frame)
            # Block Camera for not update Laf data
            camera.block()
        # Update camera counter
        camera.counter += 1

    # Kill connection with camera
    cap.release()


def Analyze_Laf_Data(
    camera: Camera,
    tasks: Queue,
    trackers: Namespace,
    configs: Namespace,
    allObjectsModel: TensorModel,
    trafficLightModel: TensorModel
) -> None:
    # Wait until connection to the camera
    print(f'The analysis task is waiting to connection to the camera: {camera}')
    camera.wait()

    # If the camera is not connected, the process will be killed
    if not camera.isAlive:
        return

    # Start analyze the camera active frame
    print(f'Start analyze the camera: {camera} active frame!')
    print('#_________________________________________________________#')

    print('* _________________________________________________________ *')
    tracker: Tracker = getattr(trackers, f'{camera}')
    linesConfigs: dict = getattr(configs, f'{camera}')

    violationDetector: ViolationDetector = ViolationDetector(name=camera.ip, configs=linesConfigs)
    camera.isDetectable = True
    while True:
        if not camera.isAlive:
            # If the connection with the camera is terminated
            # The analysis process will be terminated too
            break

        if not camera.isBlocked:
            # If on Updating of Camera Last Active Frame is blocked
            # The active frame will be analyzed and after analysis, the unblocked camera's last active frame status
            continue

        print('* _________________________________________________________ *')
        # ______________________________________________________________________________
        start = time.monotonic()
        # LAF is Last Active Frame
        # Getting camera last active frame data [lafData(np.ndarray), laFQueue(int)]
        lafData, laFQueue = camera.last_active_frame_data()
        timeIt(camera=str(camera), title='Get LAF Time', startTime=start)
        # ______________________________________________________________________________

        # ______________________________________________________________________________
        start = time.monotonic()
        # Convert HWC LAF data to CHW LAF data
        chwLafData: np.ndarray = Laf_HWC_To_CHW(lafData=lafData)
        # Getting Input Tensor Data from converted CHW data
        inTenSorData: np.ndarray = CHW_Laf_To_TenSor(chwLafData=chwLafData)
        timeIt(camera=str(camera), title='Convert LAF To Time', startTime=start)
        # ______________________________________________________________________________

        while not allObjectsModel.isAsyncInferQueueReady:
            # If Model is busy or Async Infer is doing task
            # Wait for Async Infer Ready Status ready
            pass
        # ______________________________________________________________________________
        start = time.monotonic()
        # Detecting Car Boxes from LAF
        tensors: np.ndarray = allObjectsModel.detect(inTenSorData=inTenSorData)
        timeIt(camera=str(camera), title='Detect Tensors Time', startTime=start)
        # ______________________________________________________________________________

        # ______________________________________________________________________________
        start = time.monotonic()
        tlCoors, detectedCoorsOfCars, detectedCoorsOfCarPlates = Classification_Tensors(
            tensors=tensors, lafDataShape=lafData.shape)
        timeIt(camera=str(camera), title='Classificate Tensors Time', startTime=start)
        # ______________________________________________________________________________

        # If Traffic Light Is Detected In The LAF
        isTlColorUpdated: bool = False
        if tlCoors:
            # ______________________________________________________________________________
            start = time.monotonic()
            p1, p2 = tlCoors
            trafficLightImage: Image.Image = Image.fromarray(lafData[p1.y:p2.y, p1.x:p2.x])
            transformedTLImage: np.ndarray = Transform_Traffic_Light_Image(trafficLightImage=trafficLightImage)
            timeIt(camera=str(camera), title='Transform TL Time', startTime=start)
            # ______________________________________________________________________________

            # ______________________________________________________________________________
            while not trafficLightModel.isAsyncInferQueueReady:
                # If Model is busy or Async Infer is doing task
                # Wait for Async Infer Ready Status ready
                pass

            # ________________________________________________
            start = time.monotonic()
            # Detect Traffic Light From LAF Data
            trafficLight: np.ndarray = trafficLightModel.detect(inTenSorData=transformedTLImage)
            timeIt(camera=str(camera), title='Detect TL Color', startTime=start)
            # ________________________________________________

            # _________________________________________________
            start = time.monotonic()
            # Detect Color Of The Traffic Light
            tlColor: str = Traffic_Light_Color(trafficLight=trafficLight)
            timeIt(camera=str(camera), title='Get TL Color Time', startTime=start)
            # _________________________________________________

            # _________________________________________________
            start = time.monotonic()
            # Update Crossroad Traffic Light Info [Color and Coordinate]
            tracker.Update_Traffic_Light_Info(
                coors=tlCoors,
                color=tlColor,)
            timeIt(camera=str(camera), title='Updata TL Color Time', startTime=start)

            isTlColorUpdated = True
            # _________________________________________________
            # ______________________________________________________________________________

        # Update Tracker To Detected Cars And Detected Vehicle Plate Number
        if detectedCoorsOfCars:
            # ______________________________________________________________________________
            # _________________________________________________
            start = time.monotonic()
            # Update Detected Cars Coordinates and MetaData
            tracker.Update_Cars(
                detectedCoorsOfCars=detectedCoorsOfCars,
                detectedCoorsOfCarPlates=detectedCoorsOfCarPlates)
            timeIt(camera=str(camera), title='Update Detected Cars Time', startTime=start)
            # _________________________________________________

            # _________________________________________________
            start = time.monotonic()
            # Update Violations Detected Cars
            mustTasks = violationDetector.Update_Trackers(
                laFQueue=laFQueue,
                cars=tracker.cars)
            timeIt(camera=str(camera), title='Update Violations Time', startTime=start)

            start = time.monotonic()
            # Adding task to Task Queue
            for mustTask in mustTasks:
                tasks.put(mustTask)
            timeIt(camera=str(camera), title='Add Must Task To Queue Time', startTime=start)
            # _________________________________________________

            # _________________________________________________
            start = time.monotonic()
            # Checking Violations Detected Cars
            tlColor: str = tracker.traffic_light.color if isTlColorUpdated else 'None'
            violationDetector.Check_Violations(
                laFQueue=laFQueue,
                tlColor=tlColor,
                cars=tracker.cars)
            timeIt(camera=str(camera), title='Checking Violations Time', startTime=start)
            # _________________________________________________
            # ______________________________________________________________________________
        print('* _________________________________________________________ *')

        # Update Camera Blocking Status For unLock To Update Shared Memory Data
        camera.unblock()

        # if not camera.isDetectable:
        #     continue

        # for car in tracker.cars:
        #     if car.coors is None:
        #         continue
        #     p1, p2 = car.coors

        #     textPos: Point = Point(x=10, y=-10) + p1
        #     cv2.putText(
        #         img=lafData,
        #         text=f"ID:{car.id}",
        #         org=textPos.as_point(),
        #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #         fontScale=0.5,
        #         color=(0, 255, 0))
        #     cv2.rectangle(
        #         img=lafData,
        #         pt1=p1.as_point(),
        #         pt2=p2.as_point(),
        #         color=(0, 255, 0))
        # cv2.imshow(str(camera), lafData)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     camera.isDetectable = False
        #     cv2.destroyAllWindows()


def main():
    from warnings import filterwarnings
    from config import CONF
    filterwarnings('ignore')

    with Manager() as manager:
        trackers: Namespace = manager.Namespace()
        configs: Namespace = manager.Namespace()
        cameras: ListProxy[Camera] = manager.list()

        for ip in CONF['ips']:
            print('#_________________________________________________________#')
            print('        LOADING CAMERA CONFIG FILE OR DRAWING LINES        ')
            line: Line = Line(camera=ip)
            rtsp_url: str = Camera.rtsp_url(
                username=CONF["username"],
                password=CONF["password"],
                ip=ip)

            isCameraConnected, frameShape = line.load(url=rtsp_url)
            if not isCameraConnected:
                print(f'Not connection with camera: {ip}!')
                continue

            setattr(trackers, ip, Tracker(name=ip, maxDisappeared=MAX_DISAPPEARED, maxDistance=MAX_DISTANCE))
            setattr(configs, ip, line.config)
            cameras.append(
                Camera(
                    manager=manager,
                    ip=ip,
                    username=CONF["username"],
                    password=CONF["password"]))

        if not cameras:
            print('*************************************************************')
            print('* Not Found Camera Or Not Connection Any Camera To Analysis *')
            print('*************************************************************')
            return

        start = time.monotonic()
        print('Object and Traffic Light detector models are loading....')
        allObjectsModel: TensorModel = TensorModel(modelType=ModelType.ALL)
        trafficLightModel: TensorModel = TensorModel(modelType=ModelType.TRAFFIC_LIGHT)
        print('Object and Traffic Light detector models are loaded!')
        timeIt(camera=CONF['ips'], title='Loading tensor Models Time', startTime=start)
        print('#_________________________________________________________#')

        tasks: Queue = manager.Queue()  # type: ignore
        with ProcessPoolExecutor(max_workers=2) as connectionPool, \
                ThreadPoolExecutor(max_workers=12) as analyzingPool, \
        ProcessPoolExecutor(max_workers=2) as videoCreatorPool:

            connectionFutures = []
            analysisFutures = []

            for camera in camera:
                # Create Connection Task
                connectionTask = connectionPool.submit(
                    Connect_To_Camera, camera)
                connectionTask.add_done_callback(fn=Handle_Exception)
                connectionFutures.append(connectionTask)

                # Create Analyze Task
                analysisTask = analyzingPool.submit(
                    Analyze_Laf_Data, camera, tasks, trackers, configs, allObjectsModel, trafficLightModel)
                analysisTask.add_done_callback(fn=Handle_Exception)
                analysisFutures.append(analysisTask)

            # # Submit "Connection To Camera" tasks in The Connection Pool
            # connectionFutures = [
            #     connectionPool.submit(
            #         Connect_To_Camera,
            #         camera
            #     ) for camera in cameras]

            # # Submit "Analysis Laf Data" tasks in The Analysis Pool
            # analysisFutures = [
            #     analyzingPool.submit(
            #         Analyze_Laf_Data,
            #         camera,
            #         tasks,
            #         trackers,
            #         configs,
            #         allObjectsModel,
            #         trafficLightModel,
            #     ) for camera in cameras]

            # Submit "Video Creator" tasks in The Video Creator Pool
            videoCreatorFutures = [
                # videoCreatorPool.submit(
                #     Video_Creator,
                #     tasks
                # )
            ]

            # Wait for all futures to complete.
            wait(connectionFutures + analysisFutures + videoCreatorFutures)

        print('Main Thread is terminated!')
        # # Unlinking (remove) shared memory view from Memory
        for camera in cameras:
            camera.cleaning_shared_memories()


if __name__ == '__main__':
    main()
