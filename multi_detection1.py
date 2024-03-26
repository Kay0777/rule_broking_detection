import cv2
import time
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from openvino.runtime import Core
import openvino.properties as props
import openvino.properties.hint as hints
import torch.nn.functional as F
from typing import Tuple
from tracking import CentroidTracker
from ultralytics.utils import ops
from threading import Thread

ov_core = Core()

compile_config = {
    # "CPU_THREADS_NUM": "12",  # Use as many as needed
    "GPU_THROUGHPUT_STREAMS": "6",  # Or a specific number for nstreams
    #"PERF_COUNT": "YES"  # If you want to enable performance counters
}

# det_ov_model = ov_core.read_model(model='./best_openvino_model/best.xml')
det_ov_model = ov_core.read_model(model='./best_openvino_model_half/best.xml')
# device = "GPU"  # "GPU"
# det_compiled_model = ov_core.compile_model(model=det_ov_model, device_name='CPU')
det_ov_model.reshape({0: [8, 3, 640, 640]})
det_compiled_model = ov_core.compile_model(model=det_ov_model, device_name='GPU', config=compile_config)

infer_request = det_compiled_model.create_infer_request()
infer_request.start_async()

# Optimize the model
model_tl = ov_core.read_model(model='./tl_m/tl-model.xml')
compiled_model_tl = ov_core.compile_model(model=model_tl , device_name='GPU')  # or 'GPU' if available
input_layer_tl = compiled_model_tl.input(0)

tracker = {}
# tracker = CentroidTracker()

def letterbox(img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), color:Tuple[int, int, int] = (114, 114, 114), auto:bool = False, scale_fill:bool = False, scaleup:bool = False, stride:int = 32):

    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess_image(img0: np.ndarray):
    # resize
    img = letterbox(img0)[0]
    # Convert HWC to CHW
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img


def image_to_tensor(image:np.ndarray):
    input_tensor = image.astype(np.float32)  # uint8 to fp32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    # add batch dimension
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    
    return input_tensor

def postprocess(
    pred_boxes:np.ndarray,
    input_hw:Tuple[int, int],
    orig_img:np.ndarray,
    min_conf_threshold:float = 0.25,
    nms_iou_threshold:float = 0.7,
    agnosting_nms:bool = False,
    max_detections:int = 300,
    pred_masks:np.ndarray = None,
    retina_mask:bool = False
) -> list[dict]:
    nms_kwargs = {
        "agnostic": agnosting_nms,
        "max_det": max_detections
    }

    preds = ops.non_max_suppression(
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        nc=6,
        **nms_kwargs
    )
    proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

    results = []
    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "segment": []})
            continue
        if proto is None:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
            continue
        if retina_mask:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        results.append({"det": pred[:, :6].numpy(), "segment": segments})
    return results

def get_result(results: dict, label_map: dict) -> list[tuple]:
    # return [
    #     (*xyxy, conf, label_map.get(int(lbl)))
    #     for *xyxy, conf, lbl in results.get("det", [])
    # ]
    outputs = []
    boxes = results["det"]
    
    for idx, (*xyxy, conf, lbl) in enumerate(boxes):
        label = f'{label_map[int(lbl)]}'
        outputs.append((*xyxy, conf, label))
    return outputs

def handle_traffic_light_detection(image_draw: np.ndarray, bbox: tuple, frame: np.ndarray) -> str:
    # Ensure the coordinates are within the frame boundaries
    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
    
    image = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(img=Image.fromarray(frame[y1:y2, x1:x2])).numpy()
    
    
    image = np.expand_dims(image, axis=0)
    return image

def update_trackers_and_draw_ids(image_draw, car_detections, i):
    global tracker
    if i not in tracker:
        tracker[i] = CentroidTracker()

    car_tracked_objects = tracker[i].update(car_detections)

    for obj in car_tracked_objects:
        bbox = [int(x.item()) for x in obj[2]]   # bbox is in the format [x1, y1, x2, y2]
        object_id = obj[0]

        # Draw the bounding box
        cv2.rectangle(image_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Put text (ID and class) near the bounding box
        text = f"ID: {object_id}, Class: Car"
        cv2.putText(image_draw, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_draw, car_tracked_objects

def final_process(frame: np.ndarray, box: np.ndarray, corrected_tensor: np.ndarray, i:int, results: list) -> None:
    start = time.time()
    box = np.reshape(box, (1, 10, 8400))
    # print(f"Raw box values before postprocess: {box[0, :, :5]}")
    detection = postprocess(
        pred_boxes=box,
        input_hw=corrected_tensor.shape[1:],
        orig_img=frame,
        pred_masks=None
    )
    # print(f"Processed detection: {detection}")
    end = time.time()
    diff = (end - start) * 1000
    print (f"Postprocess time: {diff}")   

    start = time.time()

    rightmost_traffic_light_bbox = None
    last_known_traffic_light = None

    tl_image = None

    max_x2 = -1
    class_dict = {0: 'car', 1: 'traffic_light', 2: 'stop_sign', 3: 'plate', 4: 'allowing', 5: 'additional'}

    height, width = frame.shape[:2]
    # image_draw = frame[0:height, 0:width].copy()
    image_draw = frame
    outputs = get_result(detection[0], class_dict)
    end = time.time()
    diff = (end - start) * 1000
    print (f"Get_result time: {diff}")    
    start = time.time()
    car_detections = []
    for output in outputs:  
        bbox, score, cls = output[:4], output[4], output[5]
        if cls == "traffic_light" and score > 0.5:
            _, _, tx2, _ = [int(x.item()) for x in bbox]
            if tx2 > max_x2 and tx2 <= frame.shape[1]:
                max_x2 = tx2
                rightmost_traffic_light_bbox = [int(x.item()) for x in bbox]
        
        px1, py1, px2, py2 = (None, ) * 4
        if cls == "plate":
            px1, py1, px2, py2 = [int(x.item()) for x in bbox]

        if cls in ['car']:
            car_detections.append(bbox)
    
    # Draw the detection
    if rightmost_traffic_light_bbox is None:
        # If no traffic light is detected in this frame, use the last known position
        if last_known_traffic_light is not None:
            tl_image = handle_traffic_light_detection(image_draw, last_known_traffic_light, frame)
        else:
            tl_image = "None"
    else:
        last_known_traffic_light = rightmost_traffic_light_bbox
        tl_image = handle_traffic_light_detection(image_draw, rightmost_traffic_light_bbox, frame)

     
    # Update trackers and draw tracking IDs for all frames
    detected_objects = []
    if car_detections:
        image_draw, car_tracked_objects = update_trackers_and_draw_ids(image_draw, car_detections, i)
        for tracked_obj in car_tracked_objects:        
            track_id, _, bbox = tracked_obj  # Unpack the object ID, centroid, and bbox
            x1, y1, x2, y2 = map(int, bbox)  # Unpack the bounding box
            obj_data = {
                'id': track_id,
                'class': 'car',
                'bbox': [x1, y1, x2, y2],
                'class2': 'plate',
                'pbbox': []  # Default to empty plate bbox
            }
            # Check for a plate detection within this car's bounding box
            if all(coord is not None for coord in [px1, py1, px2, py2]):
                if x1 <= px1 <= x2 and y1 <= py1 <= y2 and x1 <= px2 <= x2 and y1 <= py2 <= y2:
                    obj_data['pbbox'] = [px1, py1, px2, py2]
            detected_objects.append(obj_data)
        # results[i].append([image_draw, detected_objects, tl_image])
    results[i] = [image_draw, detected_objects, tl_image]
    end = time.time()
    diff = (end - start) * 1000
    print (f"Finalize time: {diff}")  

def tl_color(frame: np.ndarray, analyzed_frame: np.ndarray) -> None:
    class_names = ["Green", "None", "Red", "RedGreen", "RedYellow", "RedYellowGreen", "Yellow", "YellowGreen"]
    _, x, y = frame.shape
    results = compiled_model_tl.infer_new_request({input_layer_tl.any_name: analyzed_frame})
    output = results[compiled_model_tl.output(0)]
    
    # Assuming the output is a probability distribution
    probabilities = F.softmax(torch.tensor(output), dim=1).numpy()[0]
    
    # You can then extract the top prediction and its probability
    top1_index = np.argmax(probabilities)
    top1_class_name = class_names[top1_index]
    cv2.putText(frame, f'Traffic Light: {top1_class_name}', (x - 50 , 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 0], thickness=2)
    return top1_class_name

def process_frame(frames: list[np.ndarray], tensor: list[np.ndarray]) -> list:
    corrected_tensor = np.squeeze(
        a=np.stack(tensor, axis=0),
        axis=1,
    )
    start = time.time()
    # Preprocess the image for object detection  
    infer_request.start_async(corrected_tensor)
    infer_request.wait()  
    # result = det_compiled_model(corrected_tensor)
    print (infer_request.results)
    boxes = result[det_compiled_model.output(0)]
    end = time.time()
    diff = (end - start) * 1000
    print (f"Inference time: {diff}")

    start = time.time()
    # box = np.reshape(box, (1, 10, 8400))
    # print(f"Raw box values before postprocess: {box[0, :, :5]}")
    detection = postprocess(
        pred_boxes=boxes,
        input_hw=corrected_tensor.shape[2:],
        orig_img=frames,
        pred_masks=None
    )
    # print(f"Processed detection: {detection}")
    end = time.time()
    diff = (end - start) * 1000
    print (f"Postprocess time: {diff}")   
    # # analyzed_frames: list = []
    # analyzed_frames = [None] * len(boxes)
    # threads: list = []
    # for i, box in enumerate(boxes):
    #     thread = Thread(target=final_process, args=(frames[i], box, corrected_tensor[i], i, analyzed_frames))
    #     thread.start()
    #     threads.append(thread)
    
    # for thread in threads:
    #     thread.join()
    # del threads
    # start = time.time()
    # for frame, analyzed_frame in zip(frames, analyzed_frames):
    #     if isinstance(analyzed_frame[2], str) and analyzed_frame[2] == "None":
    #         continue
    #     analyzed_frame[2] = tl_color(frame=frame, analyzed_frame=analyzed_frame[2])
    # end = time.time()
    # diff = (end - start) * 1000
    # print (analyzed_frame[2])
    # print (f"TL color time: {diff}")
    
    # return analyzed_frames
