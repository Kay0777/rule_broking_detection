
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch import from_numpy as TorchFromNP, tensor, Tensor
from concurrent.futures import Future
from ultralytics.utils import ops
import torch.nn.functional as F


from PIL import Image
import numpy as np


from kai.models import TensorModel, Point
from typing import Union

import time
import cv2

from kai.const import (
    MNS_KWARGS,
    INPUT_TENSOR_SHAPE,
    TRAFFIC_LIGHT_COLORS,
)


# Done Optimized
def Handle_Exception(future: Future) -> None:
    try:
        future.result()
    except Exception as e:
        print(e)
        exception: Exception = future.exception()
        if exception:
            print('# $ ______________________________________________________ $ #')
            print(f"Exception: {exception}")
            print('# $ ______________________________________________________ $ #')


# Done Optimized
def ReSize_Laf_Data(
        lafData: np.ndarray,
        newShape: tuple[int, int] = INPUT_TENSOR_SHAPE,
        color: tuple[int, int, int] = (114, 114, 114),
        auto: bool = False,
        scaleFill: bool = False,
        scaleUp: bool = False,
        stride: int = 32
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = lafData.shape[:2]  # current shape [height, width]
    if isinstance(newShape, int):
        newShape = (newShape, newShape)

    # Scale ratio (new / old)
    r = min(newShape[0] / shape[0], newShape[1] / shape[1])
    # Only scale down, do not scale up (for better test mAP)
    if not scaleUp:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = newShape[1] - new_unpad[0], newShape[0] - \
        new_unpad[1]  # wh padding

    # Minimum rectangle
    if auto:
        # wh padding
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    # Stretch
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (newShape[1], newShape[0])
        ratio = newShape[1] / shape[1], newShape[0] / \
            shape[0]  # width, height ratios

    # Divide padding into 2 sides
    dw /= 2
    dh /= 2

    # Resize Laf Data
    if shape[::-1] != new_unpad:
        lafData = cv2.resize(lafData, new_unpad,
                             interpolation=cv2.INTER_LINEAR)

    # Get Padding values
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Create a Copied Laf data and Add Border
    lafData = cv2.copyMakeBorder(
        lafData, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color)
    return lafData, ratio, (dw, dh)


# Done Optimized
def Laf_HWC_To_CHW(lafData: np.ndarray):
    # Resiz LAF data on HWC
    resizedLafData, *_ = ReSize_Laf_Data(lafData=lafData)

    # Convert HWC to CHW
    hwcLafData = resizedLafData.transpose(2, 0, 1)
    chwLafData = np.ascontiguousarray(hwcLafData)
    return chwLafData


# Done Optimized
def CHW_Laf_To_TenSor(chwLafData: np.ndarray) -> np.ndarray:
    # Convert LAF data dtype to float32
    inTensorData = chwLafData.astype(np.float32)  # uint8 to fp32

    # LAF Data in the range from 0 to 255
    # Convert LAF data in the range from 0.0 to 1.0
    inTensorData /= 255.0

    # Dimensional of LAF Input Data
    # If dimensional of is 3, Extract the first tensor
    if inTensorData.ndim == 3:
        inTensorData = np.expand_dims(inTensorData, 0)
    return inTensorData


# Done Optimized
def Transform_Traffic_Light_Image(trafficLightImage: Image.Image) -> np.ndarray:
    # Ensure the coordinates are within the frame boundaries
    transform: Compose = Compose([  # type: ignore
        Resize((128, 64)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    # Transform Image
    transformedImg = transform(img=trafficLightImage).numpy()
    image: np.ndarray = np.expand_dims(a=transformedImg, axis=0)
    return image


# Done Optimized
def Traffic_Light_Color(trafficLight: np.ndarray) -> str:
    probabilities = F.softmax(tensor(data=trafficLight), dim=1).numpy()[0]

    # You can then extract the top prediction and its probability
    probablyColorFirstIndex = np.argmax(probabilities)
    if probabilities[probablyColorFirstIndex] >= 0.85:
        return TRAFFIC_LIGHT_COLORS[probablyColorFirstIndex]
    return TRAFFIC_LIGHT_COLORS[1]


# Done Optimized
def Handle_Traffic_Light_Detection(trafficLightModel: TensorModel, image: np.ndarray) -> np.ndarray:
    while trafficLightModel.isAsyncInferQueueReady:
        pass

    # Detect Traffic Light From LAF Data
    trafficLight: np.ndarray = trafficLightModel.detect(image)
    return trafficLight


# Done Optimized
def Single_Post_Process(
    tensors: np.ndarray,
    lafDataShape: tuple[int, ...],
    inTenSorDataShape: tuple[int, int] = INPUT_TENSOR_SHAPE,
) -> list[Tensor]:

    pred = ops.non_max_suppression(
        prediction=TorchFromNP(tensors), **MNS_KWARGS)[0]
    pred[:, :4] = ops.scale_boxes(
        inTenSorDataShape, pred[:, :4], lafDataShape).round()
    return pred


# Done Optimized
def Classification_Tensors(
        tensors: np.ndarray,
        lafDataShape: tuple[int, ...],
    ) -> tuple[
        Union[None, tuple[Point, Point]],
        list[tuple[Point, Point]],
        list[tuple[Point, Point]],]:
    # Initialize return variables
    tlCoors: Union[None, tuple[Point, Point]] = None
    detectedCoorsOfCars: list[tuple[Point, Point]] = []
    detectedCoorsOfCarPlates: list[tuple[Point, Point]] = []

    height, width = lafDataShape[:2]
    maxX2: int = -1
    # Post Processing Part
    # Getting Tensors And Scaling Boxes
    _tensors = Single_Post_Process(tensors=tensors, lafDataShape=lafDataShape)
    for _tensor in _tensors:
        bbox, score, rule = _tensor[:-2].numpy().tolist(), float(_tensor[-2]), int(_tensor[-1])
        bbox = (
            Point(x=int(bbox[0]), y=int(bbox[1])),
            Point(x=int(bbox[2]), y=int(bbox[3])))

        # TRAFFIC LIGHT
        if rule == 1 and score > 0.5:
            _p2: Point = bbox[-1]

            if _p2.x > maxX2 and _p2.x <= width:
                maxX2 = _p2.x
                tlCoors = bbox

        # VEHICLE PLATE NUMBER
        elif rule == 3:
            detectedCoorsOfCarPlates.append(bbox)

        # CAR OBJECT
        elif rule == 0:
            detectedCoorsOfCars.append(bbox)
    return (tlCoors, detectedCoorsOfCars, detectedCoorsOfCarPlates)


if __name__ == "__main__":
    import concurrent.futures as cf
    from queue import Queue
    import numpy as np

    q = Queue()

    files = [f'data/{i}.npy' for i in range(19, 26)]
    for file in files:
        data = np.load(file)

        start = time.monotonic()
        chwLafData = Laf_HWC_To_CHW(lafData=data)
        inTenSorData = CHW_Laf_To_TenSor(chwLafData=chwLafData)
        end = time.monotonic()

        print('Diff 1: {:.3f}'.format(1000 * (end - start)))
        q.put((data, inTenSorData))
    cv2.destroyAllWindows()

    # singleObjectDetectModel: TensorModel = TensorModel(modelType=ModelType.ALL)
    # trafficLightModel: TensorModel = TensorModel(modelType=ModelType.TRAFFIC_LIGHT)
    # tracker: Tracker = Tracker(name='10.42.7.73')

    # for _ in range(5):
    #     if q.empty(): break

    #     data, tensor = q.get()
    #     if not singleObjectDetectModel.isAsyncInferQueueReady:
    #         q.put(tensor)
    #         continue

    #     boxes = singleObjectDetectModel.detect(tensor)
    #     start = time.monotonic()
    #     Detection_Real_Cars_And_TL_From_LAF(
    #         lafData=data,
    #         boxes=boxes,
    #         tracker=tracker,
    #         trafficLightModel=trafficLightModel,
    #     )
    #     end = time.monotonic()
    #     print('Diff 2: {:.3f}'.format(1000 * (end - start)))
