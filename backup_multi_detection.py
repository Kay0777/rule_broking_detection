import cv2
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from openvino.runtime import Core
import torch.nn.functional as F
from typing import Tuple, Union
from tracking import CentroidTracker
from ultralytics.utils import ops
from threading import Thread
import time
ov_core = Core()
det_ov_model = ov_core.read_model(model='./best_int8_openvino_model/best.xml')
device = "CPU"  # "GPU"
det_compiled_model = ov_core.compile_model(det_ov_model, device)

# Optimize the model
model_tl = ov_core.read_model(model='./tl_m/tl-model.xml')
compiled_model_tl = ov_core.compile_model(model=model_tl , device_name='CPU')  # or 'GPU' if available
input_layer_tl = compiled_model_tl.input(0)



def letterbox(img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), color:Tuple[int, int, int] = (114, 114, 114), auto:bool = False, scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
    """
    Resize image and padding for detection. Takes image as input,
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

    Parameters:
      img (np.ndarray): image for preprocessing
      new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
      color (Tuple(int, int, int)): color for filling padded area
      auto (bool): use dynamic input size, only padding for stride constrins applied
      scale_fill (bool): scale image to fill new_shape
      scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
      stride (int): input padding stride
    Returns:
      img (np.ndarray): image after preprocessing
      ratio (Tuple(float, float)): hight and width scaling ratio
      padding_size (Tuple(int, int)): height and width padding size


    """
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
    """
    Preprocess image according to YOLOv8 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

    Parameters:
      img0 (np.ndarray): image for preprocessing
    Returns:
      img (np.ndarray): image after preprocessing
    """
    # resize
    img = letterbox(img0)[0]

    # Convert HWC to CHW
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img


def image_to_tensor(image:np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

    Parameters:
      img (np.ndarray): image for preprocessing
    Returns:
      input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range
    """
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
    pred_masks: Union[None, np.ndarray] = None,
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

def plot_one_box(box:np.ndarray, img:np.ndarray, color:Tuple[int, int, int] = None, mask:np.ndarray = None, label:str = None, line_thickness:int = 5):
    """
    Helper function for drawing single bounding box on image
    Parameters:
        x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
        img (no.ndarray): input image
        color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
        mask (np.ndarray, *optional*, None): instance segmentation mask polygon in format [N, 2], where N - number of points in contour, if not provided, only box will be drawn
        label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
        line_thickness (int, *optional*, 5): thickness for box drawing lines
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if mask is not None:
        image_with_mask = img.copy()
        mask
        cv2.fillPoly(image_with_mask, pts=[mask.astype(int)], color=color)
        img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)
    return img


def get_result(results: dict, label_map: dict) -> list[tuple]:
    return [
        (*xyxy, conf, label_map.get(int(lbl)))
        for *xyxy, conf, lbl in results.get("det", [])
    ]


def final_process(frame: np.ndarray, box, results: list) -> list:
    
    return results

def process_frame(frames: list[np.ndarray], tensor: list[np.ndarray]) -> list:
    rightmost_traffic_light_bbox = None
    last_known_traffic_light = None

    tl_color = None

    max_x2 = -1
    class_dict = {0: 'car', 1: 'traffic_light', 2: 'stop_sign', 3: 'plate', 4: 'allowing', 5: 'additional'}

    corrected_tensor = np.squeeze(
        a=np.stack(tensor, axis=0),
        axis=1,
    )


    # Preprocess the image for object detection    
    result = det_compiled_model(corrected_tensor)
    boxes = result[det_compiled_model.output(0)]

    analyzed_frames: list = []
    start = time.time()
    detection = postprocess(
        pred_boxes=boxes,
        input_hw=corrected_tensor.shape[2:],
        orig_img=frames,
        pred_masks=None
    )
    end = time.time()
    diff = (end - start) * 1000
    print (f"Postprocess time: {diff}") 
    
    for i, det in enumerate(detection):
        height, width = frames[i].shape[:2]
        image_draw = frames[i][0:height, 0:width].copy()        
        outputs = get_result(det, class_dict)
        
        car_detections = []
        for output in outputs:  
            bbox = output[:4]  # This will be a tuple of (x1, y1, x2, y2)
            score = output[4]   # This is the confidence score
            cls = output[5]  # This is the label
            
            if cls == "traffic_light" and score > 0.5:
                # Check if this traffic light's right edge is the farthest to the right so far
                tx1, ty1, tx2, ty2 = [int(x.item()) for x in bbox]
                if tx2 > max_x2 and tx2 <= frames[i].shape[1]:
                    max_x2 = tx2
                    rightmost_traffic_light_bbox = [int(x.item()) for x in bbox]
            
            px1, py1, px2, py2 = None, None, None, None
            if cls == "plate":
                px1, py1, px2, py2 = [int(x.item()) for x in bbox]

            if cls in ['car']:
                car_detections.append(bbox)
        
        # Draw the detection
        if rightmost_traffic_light_bbox is not None:
            last_known_traffic_light = rightmost_traffic_light_bbox
            tl_color_detected = handle_traffic_light_detection(image_draw, rightmost_traffic_light_bbox, frames[i])
            if tl_color_detected:
                tl_color = tl_color_detected
        else:
            # If no traffic light is detected in this frame, use the last known position
            if last_known_traffic_light is not None:
                tl_color_detected = handle_traffic_light_detection(image_draw, last_known_traffic_light, frames[i])
                if tl_color_detected:
                    tl_color = tl_color_detected

        # Update trackers and draw tracking IDs for all frames
        if len(car_detections) > 0:
            image_draw, car_tracked_objects = update_trackers_and_draw_ids(image_draw, car_detections, i)
        else:
            car_tracked_objects = []
        detected_objects = []

        for tracked_obj in car_tracked_objects:
            track_id, centroid, bbox = tracked_obj  # Unpack the object ID, centroid, and bbox
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
        
        analyzed_frames.append([image_draw, detected_objects, tl_color])
    return analyzed_frames

def handle_traffic_light_detection(image_draw, bbox, frame):
    class_names = ["Green", "None", "Red", "RedGreen", "RedYellow", "RedYellowGreen", "Yellow", "YellowGreen"]
    transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x1, y1, x2, y2 = bbox
    # Ensure the coordinates are within the frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)  # frame.shape[1] is the width
    y2 = min(frame.shape[0], y2)  # frame.shape[0] is the height
    
    detected_traffic_light = frame[y1:y2, x1:x2]
    image = Image.fromarray(detected_traffic_light)
    image = transform(image)
    # Convert to numpy array and add batch dimension
    image = image.numpy()
    image = np.expand_dims(image, axis=0)
    results = compiled_model_tl.infer_new_request({input_layer_tl.any_name: image})
    output = results[compiled_model_tl.output(0)]
    
    # Assuming the output is a probability distribution
    probabilities = F.softmax(torch.tensor(output), dim=1).numpy()[0]
    
    # You can then extract the top prediction and its probability
    top1_index = np.argmax(probabilities)
    top1_score = probabilities[top1_index]
    
    if top1_score >= 0.85:
        top1_class_name = class_names[top1_index]
    else:
        top1_class_name = "None"

    bbox_color = (0, 255, 0)  # Green color for bounding box
    cv2.rectangle(image_draw, (x1, y1), (x2, y2), bbox_color, 2)
    cv2.putText(image_draw, f'Traffic Light: {top1_class_name}', (x2, y1 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 0], thickness=2)
    return top1_class_name

def update_trackers_and_draw_ids(image_draw, car_detections, i):
    trackers = [CentroidTracker(), CentroidTracker()]  # Create two instances of CentroidTracker

    # Use the appropriate tracker based on the value of i
    car_tracked_objects = trackers[i].update(car_detections)

    for obj in car_tracked_objects:
        bbox = [int(x.item()) for x in obj[2]]   # bbox is in the format [x1, y1, x2, y2]
        object_id = obj[0]

        # Draw the bounding box
        cv2.rectangle(image_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Put text (ID and class) near the bounding box
        text = f"ID: {object_id}, Class: Car"
        cv2.putText(image_draw, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_draw, car_tracked_objects