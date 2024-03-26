import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from openvino.runtime import Core
import torch.nn.functional as F
from typing import Tuple, Dict
import cv2
import numpy as np
from PIL import Image
from tracking import CentroidTracker
from PIL import Image
from typing import Tuple
from ultralytics.utils import ops
import torch
import openvino.properties as props
import openvino.properties.hint as hints
import numpy as np
import time
# Initialize the deepsparse pipeline
last_known_traffic_light = None

ov_core = Core()
det_ov_model = ov_core.read_model(model='./best_openvino_model_single/best.xml')
device = "CPU"  # "GPU"
det_ov_model.reshape({0: [1, 3, 640, 640]})
det_compiled_model = ov_core.compile_model(det_ov_model, device)

# Optimize the model
model_tl = ov_core.read_model(model='./tl_m/tl-model.xml')
compiled_model_tl = ov_core.compile_model(model=model_tl , device_name='CPU')  # or 'GPU' if available
input_layer_tl = compiled_model_tl.input(0)
    
CLASSES = ["car", "traffic_light", "stop_sign", "plate", "allowing", "additional"]

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
):
    """
    YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
    Parameters:
        pred_boxes (np.ndarray): model output prediction boxes
        input_hw (np.ndarray): preprocessed image
        orig_image (np.ndarray): image before preprocessing
        min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
        nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        max_detections (int, *optional*, 300):  maximum detections after NMS
        pred_masks (np.ndarray, *optional*, None): model ooutput prediction masks, if not provided only boxes will be postprocessed
        retina_mask (bool, *optional*, False): retina mask postprocessing instead of native decoding
    Returns:
       pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and segment - segmentation polygons for each element in batch
    """
    nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
    # if pred_masks is not None:
    #     nms_kwargs["nm"] = 32
    preds = ops.non_max_suppression(
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        nc=6,
        **nms_kwargs
    )
    results = []
    proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

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

def get_result(results: Dict, label_map: Dict):
    outputs = []
    for _, (*xyxy, conf, lbl) in enumerate(results["det"]):
        label = f'{label_map[int(lbl)]}'
        outputs.append((*xyxy, conf, label))
    return outputs


def processing_frame(frames, tensor):
    rightmost_traffic_light_bbox = None
    last_known_traffic_light = None
    tl_color = None
    rightmost_traffic_light_bbox = None
    max_x2 = -1
    class_dict = {0: 'car', 1: 'traffic_light', 2: 'stop_sign', 3: 'plate', 4: 'allowing', 5: 'additional'}
    # Increment the frame counter
    num_outputs = len(det_compiled_model.outputs)
    # Preprocess the image for object detection
    start = time.time()
    result = det_compiled_model(tensor)
    boxes = result[det_compiled_model.output(0)]
    
    end = time.time()
    diff = (end - start) * 1000
    print (f"Inference time: {diff}")
    masks = None
    input_hw = tensor[0].shape[2:]
    # print (input_hw)
    # print (frames.shape)
    # for i, frame in enumerate(frames[0]):
    
    detection = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=frames, pred_masks=masks)
    car_detections = []
    height, width, _ = frames[0].shape
    outputs = get_result(detection[0], class_dict)
    print (outputs)
    for output in outputs:
        bbox = output[:4]  # This will be a tuple of (x1, y1, x2, y2)
        score = output[4]   # This is the confidence score
        cls = output[5]  # This is the label

        if cls == "traffic_light" and score > 0.5:
            # Check if this traffic light's right edge is the farthest to the right so far
            tx1, ty1, tx2, ty2 = [int(x.item()) for x in bbox]
            if tx2 > max_x2 and tx2 <= frames[0].shape[1]:
                max_x2 = tx2
                rightmost_traffic_light_bbox = [int(x.item()) for x in bbox]
        
        px1, py1, px2, py2 = None, None, None, None
        if cls == "plate":
            px1, py1, px2, py2 = [int(x.item()) for x in bbox]

        if cls in ['car']:
            car_detections.append(bbox)
    
    # Draw the detection
    image_draw = frames[0][0:height, 0:width].copy()
    if rightmost_traffic_light_bbox is not None:
        last_known_traffic_light = rightmost_traffic_light_bbox
        tl_color_detected = handle_traffic_light_detection(image_draw, rightmost_traffic_light_bbox, frames)
        if tl_color_detected:
            tl_color = tl_color_detected
    else:
        # If no traffic light is detected in this frame, use the last known position
        if last_known_traffic_light is not None:
            tl_color_detected = handle_traffic_light_detection(image_draw, last_known_traffic_light, frames)
            if tl_color_detected:
                tl_color = tl_color_detected

    # Update trackers and draw tracking IDs for all frames
    if len(car_detections) > 0:
        image_draw, car_tracked_objects = update_trackers_and_draw_ids(image_draw, car_detections)
    else:
        car_tracked_objects = []
    
    detected_objects = []
    analyzed_frame = []
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
    analyzed_frame.append([image_draw, detected_objects, tl_color])
    return analyzed_frame

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

def update_trackers_and_draw_ids(image_draw, car_detections):
    new_tracker = CentroidTracker()

    # Use the appropriate tracker based on the value of i
    car_tracked_objects = new_tracker.update(car_detections)

    for obj in car_tracked_objects:
        bbox = [int(x.item()) for x in obj[2]]   # bbox is in the format [x1, y1, x2, y2]
        object_id = obj[0]

        # Draw the bounding box
        cv2.rectangle(image_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Put text (ID and class) near the bounding box
        text = f"ID: {object_id}, Class: Car"
        cv2.putText(image_draw, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_draw, car_tracked_objects


def main():
    from analysis import Laf_HWC_To_CHW, CHW_Laf_To_TenSor, SingleTenSorModel
    from queue import Queue

    files = ['frames/10.42.7.72/43.npy']
    # tensors = [np.load(file) for file in files]

    q = Queue()
    for file in files:
        data = np.load(file)

        start = time.monotonic()
        chwLafData = Laf_HWC_To_CHW(lafData=data)
        inTenSorData = CHW_Laf_To_TenSor(chwLafData=chwLafData)
        end = time.monotonic()

        print('Diff: {:.3f}'.format(1000 * (end - start)))
        q.put((data, inTenSorData))
    
    model: SingleTenSorModel = SingleTenSorModel()
    for _ in range(5):
        if q.empty(): break
        
        data, tensor = q.get()
        if model.isAsyncInferQueueReady:
            boxes = model.detect(tensor)
            if boxes is None:
                q.put(tensor)
                continue
            
            # print(boxes)
            processing_frame(data, boxes)
        else:
            q.put(tensor)



if __name__=="__main__":
    main()