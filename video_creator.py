from numpy import ndarray as NdArray, linalg
from numpy import (
    array as NpArray,
    load as LoadNpy,
    all as NpAll,
    floor as NpFloor,
    ceil as NpCeil,
    copy
)

from kai.models import Point
from typing import Any, Union
from turbojpeg import TurboJPEG

from config import CONF
import time
import os

import cv2

FPS: int = 10
SECONDS: int = 6
WAIT_FRAME_COUNT: int = 3

VIDEO_SHAPE: tuple[int, int] = (1280, 720)
CAR_COLOR: tuple[int, int, int] = (0, 255, 0)
CAR_PLATE_COLOR: tuple[int, int, int] = (255, 0, 0)
VIOLATION_TYPE_TO_ID: dict[str, int] = {
    'no_violation': 0,
    'red_light': 39,
    'incorrect_direction': 42,
    '1.1_line': 90,
    'stop_line': 101
}


def Transformer_New_Task_Schema_To_Old_Task_Schema(task) -> Any:
    camera, lostCarID, laFQueue, trackedCarInfo, carViolations, farthest_1_1_Line = task
    p1, p2 = farthest_1_1_Line

    _pbboxes = []
    for box in trackedCarInfo['pbboxes']:
        if not box:
            p1, p2 = box
            _pbboxes.append([p1.x, p1.y, p2.x, p2.y])
        else:
            _pbboxes.append(box)
    trackedCarInfo['pbboxes'] = _pbboxes

    _bboxes = []
    for box in trackedCarInfo['pbboxes']:
        if not box:
            p1, p2 = box
            _bboxes.append([p1.x, p1.y, p2.x, p2.y])
        else:
            _bboxes.append(box)
    trackedCarInfo['bboxes'] = _bboxes

    sourcePath: str = os.path.join(CONF['path'], CONF['main_foldername'], CONF['video_and_images_save_foldername'])
    os.makedirs(name=sourcePath, exist_ok=True)

    return {
        "obj_id": lostCarID,
        "frame_id": laFQueue,
        "ip_address": camera,
        "tracker": trackedCarInfo,
        "has_violation": True,
        "detected_violations": {
            lostCarID: carViolations
        },
        "violation_video_path": sourcePath,
        "farthest_line": [
            [p1.x, p1.y],
            [p2.x, p2.y]
        ],
    }


def Create_Range_Of_Frames(detectedFrameID: int) -> range:
    countOfFrames = FPS * SECONDS
    if detectedFrameID <= countOfFrames // 2:
        return range(1, countOfFrames + 1)

    fromFrameID: int = detectedFrameID - countOfFrames // 2
    toFrameID: int = 1 + detectedFrameID + countOfFrames // 2

    return range(fromFrameID, toFrameID)


def Wait_Untill_Frame_Is_Exists(camera: str, toFrameID: int) -> None:
    waitingPathForFrame: str = os.path.join(CONF['path'], CONF['foldername'], camera, f'{toFrameID}.npy')
    while not os.path.exists(path=waitingPathForFrame):
        pass


def Create_Car_Video_And_Image_Paths(camera: str, lostCarID: int, vtype: str) -> tuple[str, str]:
    current_time: str = time.strftime("%Y.%m.%d-%H.%M.%S")
    violationID: int = VIOLATION_TYPE_TO_ID[vtype]

    mainPath: str = os.path.join(CONF['path'], CONF['main_foldername'], CONF['video_and_images_save_foldername'])
    os.makedirs(mainPath, exist_ok=True)

    videoFilename: str = f"{current_time}__carID:{lostCarID}_RuleID:{violationID}__{camera}_video.mp4"
    videoOutputFilePath: str = os.path.join(mainPath, videoFilename)

    fullImageFilename: str = f"{current_time}__carID:{lostCarID}_RuleID:{violationID}__{camera}_full.jpeg"
    fullImageOutputFilePath: str = os.path.join(mainPath, fullImageFilename)

    carImageFilename: str = f"{current_time}__carID:{lostCarID}_RuleID:{violationID}__{camera}_car.jpeg"
    carImageOutputFilePath: str = os.path.join(mainPath, carImageFilename)

    plateImageFilename: str = f"{current_time}__carID:{lostCarID}_RuleID:{violationID}__{camera}_plate.jpeg"
    plateImageOutputFilePath: str = os.path.join(mainPath, plateImageFilename)

    return videoOutputFilePath, fullImageOutputFilePath, carImageOutputFilePath, plateImageOutputFilePath


def Create_Car_Image_Path(camera: str, lostCarID: int, vtype: str = "no_violation") -> tuple[str, str]:
    current_time: str = time.strftime("%Y.%m.%d-%H.%M.%S")
    violationID: int = VIOLATION_TYPE_TO_ID[vtype]

    mainPath: str = os.path.join(CONF['path'], CONF['main_foldername'], CONF['only_image_save_foldername'])
    os.makedirs(mainPath, exist_ok=True)

    fullImageFilename: str = f"{current_time}__carID:{lostCarID}_RuleID:{violationID}__{camera}_full.jpeg"
    fullImageOutputFilePath: str = os.path.join(mainPath, fullImageFilename)

    carImageFilename: str = f"{current_time}__carID:{lostCarID}_RuleID:{violationID}__{camera}_car.jpeg"
    carImageOutputFilePath: str = os.path.join(mainPath, carImageFilename)

    return fullImageOutputFilePath, carImageOutputFilePath


def Create_Image(inputdata: NdArray, outputFilename: str) -> None:
    engine: TurboJPEG = TurboJPEG()
    with open(file=outputFilename, mode='wb') as file:
        file.write(engine.encode(img_array=inputdata, quality=100))
        file.close()


def Scale_Coordinate(coor: tuple[Point, Point], originalShape: tuple[int, int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    Dx = VIDEO_SHAPE[0] / originalShape[1]
    Dy = VIDEO_SHAPE[1] / originalShape[0]

    p1, p2 = coor
    return (
        (int(p1.x * Dx), int(p1.y * Dy)),
        (int(p2.x * Dx), int(p2.y * Dy))
    )


def Delta_Coordinate_Car_Plate(coor: Union[None, tuple[Point, Point]], index: int, carCoors: list[Point], plateCoors: list[Point]) -> tuple[Point, Point]:
    if coor is not None:
        return coor

    detectPlateIndex: int = -1
    for i in range(index, -1, -1):
        if plateCoors[i] is not None:
            detectPlateIndex = i
            break

    if detectPlateIndex == -1:
        for i, coor in enumerate(plateCoors[index:]):
            if coor is not None:
                detectPlateIndex = index + i
                break

    carCoor1: tuple[Point, Point] = carCoors[detectPlateIndex]
    carCoor2: tuple[Point, Point] = carCoors[index]

    p1, _ = carCoor1
    p2, _ = carCoor2

    Dx = p2.x - p1.x
    Dy = p2.y - p1.y

    Dx = int(NpCeil(Dx)) if Dx > 0 else int(NpFloor(Dx))
    Dy = int(NpCeil(Dy)) if Dy > 0 else int(NpFloor(Dy))

    plateCoor: tuple[Point, Point] = plateCoors[detectPlateIndex]
    return (
        plateCoor[0] + Point(x=Dx, y=Dy),
        plateCoor[1] + Point(x=Dx, y=Dy),
    )


def Create_Video(task: tuple[str, int, dict, set]) -> None:
    camera, lostCarID, hasTheCarBrokenAnyRules, isTheCarPlateDetecedAnyTime, trackedCarInfo, carViolations = task

    if not hasTheCarBrokenAnyRules:
        # If not detected broked rules we need save only car image
        return

    if not isTheCarPlateDetecedAnyTime:
        # If the car plate not detected any time
        # The car video will not create
        return

    if 'incorrect_direction' in carViolations:
        vtype = 'incorrect_direction'
    elif 'red_light' in carViolations:
        vtype = 'red_light'
    elif 'stop_line' in carViolations:
        vtype = 'stop_line'
    elif '1.1_line' in carViolations:
        vtype = '1.1_line'
    else:
        return

    detectedFrameID = 0
    for _vtype, violationFrameID in trackedCarInfo['violation_frame']:
        if _vtype == vtype:
            detectedFrameID = violationFrameID
            break

    videoOutputFilePath, fullImageOutputFilePath, carImageOutputFilePath, plateImageOutputFilePath = Create_Car_Video_And_Image_Paths(
        camera=camera,
        lostCarID=lostCarID,
        vtype=vtype)

    rangeFrames: range = Create_Range_Of_Frames(detectedFrameID=detectedFrameID)
    Wait_Untill_Frame_Is_Exists(camera=camera, toFrameID=rangeFrames[-1])

    # $ => _____________________________________________________ <= $ #
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(
        filename=videoOutputFilePath,
        fourcc=fourcc,
        fps=FPS,
        frameSize=VIDEO_SHAPE)

    for frameID in rangeFrames:
        frameFilename = os.path.join(CONF['path'], CONF['foldername'], camera, f'{frameID}.npy')

        data: NdArray = LoadNpy(file=frameFilename)
        _data: NpArray = copy(data)
        image = cv2.resize(src=data, dsize=VIDEO_SHAPE)

        videoWriter.write(image=image)
        if detectedFrameID != frameID:
            continue

        # saving car image
        frameIndices: list[int] = trackedCarInfo['frame_indices']
        index: int = frameIndices.index(detectedFrameID)

        carCoor: tuple[Point, Point] = trackedCarInfo['bboxes'][index]
        # scaledCarCoor => (x1, y1, x2, y2)
        pt1, pt2 = Scale_Coordinate(coor=carCoor, originalShape=data.shape)
        cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=CAR_COLOR, thickness=2, lineType=2)
        carPlateCoor: Union[None, tuple[Point, Point]] = trackedCarInfo['pbboxes'][index]

        deltaCarPlateCoor: tuple[Point, Point] = Delta_Coordinate_Car_Plate(
            coor=carPlateCoor,
            index=index,
            carCoors=trackedCarInfo['bboxes'],
            plateCoors=trackedCarInfo['pbboxes'])

        pt1, pt2 = Scale_Coordinate(coor=deltaCarPlateCoor, originalShape=data.shape)
        cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=CAR_PLATE_COLOR, thickness=2, lineType=2)

        # Create a Full Image
        Create_Image(inputdata=image, outputFilename=fullImageOutputFilePath)

        # Create Full Car Image
        p1, p2 = carCoor[0]['coor']
        carImg = _data[p2.y:p1.y, p1.x:p2.x, :]
        Create_Image(inputdata=carImg, outputFilename=carImageOutputFilePath)

        # Create Full Plate Image
        p1, p2 = deltaCarPlateCoor
        plateImg = _data[p1.y:p2.y, p1.x:p2.x, :]
        Create_Image(inputdata=plateImg, outputFilename=plateImageOutputFilePath)

    if videoWriter.isOpened():
        videoWriter.release()
        cv2.destroyAllWindows()

    # tempVideoOutputFilePath = videoOutputFilePath.replace('.mp4', '_temp.mp4')
    # conversionCMD = f"ffmpeg -y -i {videoOutputFilePath} -vcodec libx264 {tempVideoOutputFilePath} >/dev/null 2>&1"
    # os.system(conversionCMD)
    # time.sleep(3)
    # if os.path.exists(tempVideoOutputFilePath):
    #     os.replace(tempVideoOutputFilePath, videoOutputFilePath)
    # # $ => _____________________________________________________ <= $ #


def Create_Video2(task: tuple[str, int, dict, set]) -> None:
    camera, lostCarID, hasTheCarBrokenAnyRules, isTheCarPlateDetecedAnyTime, trackedCarInfo, carViolations = task
    if not hasTheCarBrokenAnyRules:
        # If not detected broked rules we need save only car image
        return

    if not isTheCarPlateDetecedAnyTime:
        # If the car plate not detected any time
        # The car video will not create
        return

    if 'incorrect_direction' in carViolations:
        vtype = 'incorrect_direction'
    elif 'red_light' in carViolations:
        vtype = 'red_light'
    elif 'stop_line' in carViolations:
        vtype = 'stop_line'
    elif '1.1_line' in carViolations:
        vtype = '1.1_line'
    else:
        return

    detectedFrameID = 0
    for _vtype, violationFrameID in trackedCarInfo['violation_frame']:
        if _vtype == vtype:
            detectedFrameID = violationFrameID
            break

    rangeFrames: range = Create_Range_Of_Frames(detectedFrameID=detectedFrameID)

    videoOutputFilePath, fullImageOutputFilePath, carImageOutputFilePath, plateImageOutputFilePath = Create_Car_Video_And_Image_Paths(
        camera=camera,
        lostCarID=lostCarID,
        vtype=vtype)
    # Wait_Untill_Frame_Is_Exists(camera=camera, toFrameID=rangeFrames[-1])
    # $ => _____________________________________________________ <= $ #
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(
        filename=videoOutputFilePath,
        fourcc=fourcc,
        fps=FPS,
        frameSize=VIDEO_SHAPE)

    rangeFrames = range(330, 401)
    for frameID in rangeFrames:
        frameFilename = os.path.join(CONF['path'], CONF['foldername'], camera, f'{frameID}.npy')
        data: NdArray = LoadNpy(file=frameFilename)
        _data: NpArray = copy(data)

        carCoor = list(filter(lambda x: x['index'] == frameID, trackedCarInfo['bboxes']))
        if len(carCoor) != 0:
            pt1, pt2 = carCoor[0]['coor']
            cv2.rectangle(img=data, pt1=pt1.as_point(), pt2=pt2.as_point(), color=CAR_PLATE_COLOR, thickness=2, lineType=2)

        plateCoor = list(filter(lambda x: x['index'] == frameID, trackedCarInfo['pbboxes']))
        if len(plateCoor) != 0:
            pt1, pt2 = plateCoor[0]['coor']
            cv2.rectangle(img=data, pt1=pt1.as_point(), pt2=pt2.as_point(), color=CAR_PLATE_COLOR, thickness=2, lineType=2)

        image = cv2.resize(src=data, dsize=VIDEO_SHAPE)
        videoWriter.write(image=image)

        if detectedFrameID != frameID:
            continue

        # Create Full Image
        Create_Image(inputdata=image, outputFilename=fullImageOutputFilePath)

        # Create Full Car Image
        p1, p2 = carCoor[0]['coor']
        carImg = _data[p2.y:p1.y, p1.x:p2.x, :]
        Create_Image(inputdata=carImg, outputFilename=carImageOutputFilePath)

        # Create Full Plate Image
        p1, p2 = plateCoor[0]['coor']
        plateImg = _data[p1.y:p2.y, p1.x:p2.x, :]
        Create_Image(inputdata=plateImg, outputFilename=plateImageOutputFilePath)

    if videoWriter.isOpened():
        videoWriter.release()
        cv2.destroyAllWindows()


def read_file(file: str) -> list:
    import json

    with open(file=file, mode='r') as f:
        data = json.loads(s=f.read())
        f.close()
    return data


def main():
    import os
    import numpy as np
    from kai.models import Point

    # task => tuple[int, int, dict, dict]

    camera = '10.42.7.72'
    lostCarID = 1
    trackedCarInfo = {
        'first_seen': 330,
        'last_seen': 400,
        'frame_indices': [330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349],
        'bboxes': [],
        'pbboxes': [],
        'missing_frames': 6,
        'violation_frame': [('1.1_line', 342), ],
    }
    carViolations = set({'1.1_line'})

    task = (camera, lostCarID, trackedCarInfo, carViolations)
    # Create_Video(task=task)
    data = read_file('car.json')
    _data1 = []
    for coor in data:
        c1, c2 = coor['coors']

        p1 = Point(x=c1[0], y=c1[1])
        p2 = Point(x=c2[0], y=c2[1])
        _data1.append({"index": coor['index'], "coor": (p1, p2)})

    data = read_file('plate.json')
    _data2 = []
    for coor in data:
        c1, c2 = coor['coors']

        p1 = Point(x=c1[0], y=c1[1])
        p2 = Point(x=c2[0], y=c2[1])
        _data2.append({"index": coor['index'], "coor": (p1, p2)})

    trackedCarInfo['bboxes'] = _data1
    trackedCarInfo['pbboxes'] = _data2
    Create_Video2(task=task)


if __name__ == "__main__":
    main()

    # print([i for i in range(330, 350)])
