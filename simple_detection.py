import cv2
import time
import io
import numpy as np
import torch
from ultralytics import YOLO
from deepsparse import Pipeline
from torchvision import transforms
import torch.nn.functional as F
from openvino.runtime import Core
#from sort.sort import Sort
from shapely.geometry import Point, Polygon
from tracking_copy import CentroidTracker
from PIL import Image
# Initialize the deepsparse pipeline
last_known_traffic_light = None
device = torch.device("cpu")
ov_model = YOLO('best_int8_openvino_model/')
ov_core = Core()

# Optimize the model
model_tl = ov_core.read_model(model='./tl_m/tl-model.xml')
compiled_model_tl = ov_core.compile_model(model=model_tl , device_name='CPU')  # or 'GPU' if available
input_layer_tl = compiled_model_tl.input(0)
    
CLASSES = ["car", "traffic_light", "stop_sign", "plate", "allowing", "additional"]

def is_inside_roi(bbox, rois):
    px1, py1, px2, py2 = bbox.xyxy[0]
    detection_center = Point((px1 + px2) / 2, (py1 + py2) / 2)
    for roi in rois:
        # roi is expected to be a list of points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        roi_polygon = Polygon(roi)
        if roi_polygon.contains(detection_center):
            return True
    return False

frame_counter = 0
new_tracker = CentroidTracker()
def process_frame(frame, rois):
    global frame_counter
    global last_known_traffic_light
    tl_color = None
    class_dict = {0: 'car', 1: 'traffic_light', 2: 'stop_sign', 3: 'plate', 4: 'allowing', 5: 'additional'}
    # Increment the frame counter
    frame_counter += 1

    # Preprocess the image for object detection
    results = ov_model(frame)

    # Extract detection data
    
    # Apply Non-Max Suppression
    height, width = frame.shape[:2]
    image_draw = frame[0:height, 0:width].copy()
    car_detections = []

    rightmost_traffic_light_bbox = None
    max_x2 = -1
    for result in results:
        boxes = result.boxes.cpu()
        for bbox in boxes:
            score = bbox.conf.item()
            label = int(bbox.cls.item())
            cls = class_dict[label]
            # Scale the bounding box back to the original frame size
            if cls == "traffic_light" and score > 0.5:
                # Check if this traffic light's right edge is the farthest to the right so far
                tx1, ty1, tx2, ty2 = bbox.xyxy[0]
                if tx2 > max_x2 and tx2 <= frame.shape[1]:
                    max_x2 = tx2
                    rightmost_traffic_light_bbox = bbox.xyxy[0]
            
            px1, py1, px2, py2 = None, None, None, None
            if cls == "plate":
                px1, py1, px2, py2 = bbox.xyxy[0].int().tolist()

            if cls in ['car'] and is_inside_roi(bbox, rois):
                detection_for_tracking = bbox.xyxy[0].tolist()
                car_detections.append(detection_for_tracking)
        # Draw the detection
    if rightmost_traffic_light_bbox is not None:
        last_known_traffic_light = rightmost_traffic_light_bbox
        tl_color_detected = handle_traffic_light_detection(image_draw, rightmost_traffic_light_bbox, frame)
        if tl_color_detected:
            tl_color = tl_color_detected
    else:
        # If no traffic light is detected in this frame, use the last known position
        if last_known_traffic_light is not None:
            tl_color_detected = handle_traffic_light_detection(image_draw, last_known_traffic_light, frame)
            if tl_color_detected:
                tl_color = tl_color_detected
    # Update trackers and draw tracking IDs for all frames
    if len(car_detections) > 0:
        image_draw, car_tracked_objects = update_trackers_and_draw_ids(image_draw, car_detections)
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
    return image_draw, detected_objects, tl_color

def handle_traffic_light_detection(image_draw, bbox, frame):
    class_names = ["Green", "None", "Red", "RedGreen", "RedYellow", "RedYellowGreen", "Yellow", "YellowGreen"]
    transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x1, y1, x2, y2 = bbox.int().tolist()
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
    # Check if car_detections is not empty before updating the tracker
    car_tracked_objects = new_tracker.update(car_detections)
    return image_draw, car_tracked_objects