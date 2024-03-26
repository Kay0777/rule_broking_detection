import os
import cv2
import time
import logging
import concurrent.futures
from turbojpeg import TurboJPEG

lib_path = "/home/chorraxa/.venv310/lib/python3.10/site-packages/turbojpeg.py"
lib_path = "./libjpeg-turbo-3.0.2"
engine = TurboJPEG(lib_path=lib_path)

logging.basicConfig(filename='log.txt', level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')

FRAMES_PER_SECOND = 10
VIOLATION_TYPE_TO_ID = {
    'no_violation': 0,
    'red_light': 39,
    'incorrect_direction': 42,
    '1.1_line': 90,
    'stop_line': 101
}

frames_directory = "./frames"


def load_frame_from_jpeg(frame_idx, frames_directory, ip_address, max_retries=1):
    frame_file_path = os.path.join(frames_directory, ip_address, f"{frame_idx}.jpg")
    for _ in range(max_retries):
        if os.path.exists(frame_file_path):
            frame = cv2.imread(frame_file_path)
            if frame is not None:
                return frame
            else:
                return None
        time.sleep(1)
    return None


def save_violation_video_if_needed(obj_id, frame_id, ip_address, tracker, has_violation, detected_violations, violation_video_path, farthest_line):
    print(f"Debug: Checking if video needs to be saved for obj_id {obj_id}")
    current_time = time.strftime("%Y%m%d_%H%M%S")
    time.sleep(3)
    if has_violation:

        start_frame_index = tracker['frame_indices'][0]
        eligible_frame_indices = [idx for idx in tracker['frame_indices'] if idx >= start_frame_index + 1]

        has_pbbox = False
        for frame_idx in eligible_frame_indices:
            pbbox_idx = tracker['frame_indices'].index(frame_idx)
            pbbox = tracker['pbboxes'][pbbox_idx]
            if pbbox and len(pbbox) == 4:
                has_pbbox = True
                break

        if has_pbbox:
            violation_types = detected_violations.get(obj_id, set())
            if 'incorrect_direction' in violation_types:
                violation_type = 'incorrect_direction'
            elif 'red_light' in violation_types:
                violation_type = 'red_light'
            elif 'stop_line' in violation_types:
                violation_type = 'stop_line'
            elif '1.1_line' in violation_types:
                violation_type = '1.1_line'

            # for violation_type in violation_types:
            violation_id = VIOLATION_TYPE_TO_ID[violation_type]
            # Retrieve the corresponding frame_id for the violation_type
            frame_id = None
            for vt, fid in tracker['violation_frame']:
                if vt == violation_type:
                    frame_id = fid
                    break  # Exit the loop once the matching frame_id is found)
            save_violation_video(obj_id, tracker, ip_address, violation_id, frame_id, current_time, violation_video_path)
            save_object_image(obj_id, tracker, ip_address, violation_id, current_time, violation_video_path, save_full_frame=False)
        else:
            print("No plate was detected")
    else:
        violation_id = 0
        box = tracker['bboxes'][~0]
        lx1 = farthest_line[0][0]  # y1 from the first point
        ly1 = farthest_line[0][1]  # y1 from the first point
        lx2 = farthest_line[1][0]  # y1 from the first point
        ly2 = farthest_line[1][1]  # y2 from the second point
        m = (ly2 - ly1)/(lx2 - lx1)
        b = ly1 - m * lx1
        x = (box[3] - b) / m
        if x < (box[2] * 0.2):
            save_object_image(obj_id, tracker, ip_address, violation_id, current_time, violation_video_path)


def save_object_image(obj_id, tracker, ip_address, violation_id, current_time, violation_video_path, save_full_frame=True):
    # Use the provided save_path or default to violation_image_path
    save_path = violation_video_path
    start_frame_index = tracker['frame_indices'][0]
    eligible_frame_indices = [idx for idx in tracker['frame_indices'] if idx >= start_frame_index + 1]
    # Find the frame with the largest bbox
    largest_bbox_area = 0
    frame_with_largest_bbox = None
    bbox_for_largest_frame = None
    has_pbbox = False

    for frame_idx in eligible_frame_indices:
        pbbox_idx = tracker['frame_indices'].index(frame_idx)
        pbbox = tracker['pbboxes'][pbbox_idx]
        if pbbox and len(pbbox) == 4:
            has_pbbox = True
            break

    for frame_idx in eligible_frame_indices:
        bbox_idx = tracker['frame_indices'].index(frame_idx)
        pbbox = tracker['pbboxes'][bbox_idx]
        bbox = tracker['bboxes'][bbox_idx]

        if has_pbbox and pbbox and len(pbbox) == 4:
            target_bbox = pbbox
            bbox_area = target_bbox[1]
            if bbox_area > largest_bbox_area:
                largest_bbox_area = bbox_area
                frame_with_largest_bbox = frame_idx
                bbox_idx = tracker['frame_indices'].index(frame_with_largest_bbox)
                bbox = tracker['bboxes'][bbox_idx]
                bbox_for_largest_frame = bbox
                pbbox = tracker['pbboxes'][bbox_idx]
                pbbox_for_largest_frame = pbbox
        elif not has_pbbox and bbox and len(bbox) == 4:
            target_bbox = bbox
            bbox_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
            if bbox_area > largest_bbox_area:
                largest_bbox_area = bbox_area
                frame_with_largest_bbox = frame_idx
                bbox_idx = tracker['frame_indices'].index(frame_with_largest_bbox)
                bbox = tracker['bboxes'][bbox_idx]
                bbox_for_largest_frame = bbox
                pbbox = tracker['pbboxes'][bbox_idx]
                pbbox_for_largest_frame = pbbox
        else:
            continue

    if frame_with_largest_bbox is None or bbox_for_largest_frame is None:
        print("No eligble box")
        return

    frame = load_frame_from_jpeg(frame_with_largest_bbox, frames_directory, ip_address)
    if frame is None:
        return

        # Extract the object from the frame
    object_img = frame[bbox_for_largest_frame[1]:bbox_for_largest_frame[3], bbox_for_largest_frame[0]:bbox_for_largest_frame[2]]

    if pbbox_for_largest_frame is not None and len(pbbox_for_largest_frame) == 4:
        x1, y1, x2, y2 = pbbox_for_largest_frame

        # Decrease x1, y1 by 10%
        x1 = int(x1 * 0.99)
        y1 = int(y1 * 0.99)

        # Increase x2, y2 by 10%
        x2 = int(x2 * 1.01)
        y2 = int(y2 * 1.01)

        # Ensure that the coordinates do not exceed the frame's dimensions
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])

        # Use the modified coordinates to extract the plate image
        plate_img = frame[y1:y2, x1:x2]
    else:
        print("No eligible box for plate")
        plate_img = None

    filename = f"{current_time}_{obj_id}_{violation_id}_{ip_address}_car.jpeg"
    xmlname = f"{current_time}_{obj_id}_{violation_id}_{ip_address}.xml"
    pfilename = f"{current_time}_{obj_id}_{violation_id}_{ip_address}_plate.jpeg"

    file_path = os.path.join(save_path, filename)
    xml_path = os.path.join(save_path, xmlname)
    pfile_path = os.path.join(save_path, pfilename)

    if not os.path.exists(violation_video_path):
        print("Debug: Creating directory for object images")
        os.makedirs(violation_video_path)

    if save_full_frame:
        full_frame_filename = f"{current_time}_{obj_id}_{violation_id}_{ip_address}_full.jpeg"
        full_frame_file_path = os.path.join(save_path, full_frame_filename)
        resized_frame = cv2.resize(frame, (1280, 720))
        # Draw only the bounding box for the largest area on the resized frame
        if bbox_for_largest_frame:
            # Scale the bounding box coordinates for the resized frame
            scale_x, scale_y = 1280 / frame.shape[1], 720 / frame.shape[0]
            scaled_bbox = [int(coord * scale_x if i % 2 == 0 else coord * scale_y) for i, coord in enumerate(bbox_for_largest_frame)]

            # Draw the bounding box on the resized frame
            cv2.rectangle(resized_frame, (scaled_bbox[0], scaled_bbox[1]), (scaled_bbox[2], scaled_bbox[3]), (0, 255, 0), 2)

            with open(full_frame_file_path, 'wb') as file:
                file.write(engine.encode(resized_frame, quality=90))
                file.close()
            print("Debug: Resized full frame image with bbox saved successfully")
    if object_img is not None:
        if object_img.size == 0:
            print("Error: 'object_img' is empty")
        else:
            with open(file_path, 'wb') as file:
                file.write(engine.encode(object_img, quality=100))
                file.close()
            with open(xml_path, 'w') as xml_file:
                xml_file.write("")
    if plate_img is not None:
        with open(pfile_path, 'wb') as file:
            file.write(engine.encode(plate_img, quality=100))
            file.close()
    print("Debug: Image saved successfully")


def interpolate_bboxes(boxes):
    sorted_frames = sorted(boxes.keys())
    for i in range(len(sorted_frames) - 1):
        start_frame = sorted_frames[i]
        end_frame = sorted_frames[i + 1]

        # Calculate the difference between frames
        frame_diff = end_frame - start_frame

        if frame_diff > 1:
            start_bbox = boxes[start_frame]
            end_bbox = boxes[end_frame]

            # Linearly interpolate bbox coordinates for each frame in the gap
            for j in range(1, frame_diff):
                interp_bbox = []
                for k in range(len(start_bbox)):
                    # Linear interpolation for each coordinate
                    interp_value = start_bbox[k] + (end_bbox[k] - start_bbox[k]) * j / frame_diff
                    interp_bbox.append(int(interp_value))

                # Assign the interpolated bbox to the corresponding frame
                boxes[start_frame + j] = interp_bbox
    return boxes


def save_violation_video(obj_id, tracker, ip_address, violation_id, violation_frame_id, current_time, violation_video_path):
    def scale_bbox(bbox, original_size, target_size):
        x_scale = target_size[0] / original_size[0]
        y_scale = target_size[1] / original_size[1]
        return [int(bbox[0] * x_scale), int(bbox[1] * y_scale), int(bbox[2] * x_scale), int(bbox[3] * y_scale)]
    # Validate frame range
    # Define paths and filenames
    video_filename = f"{current_time}_{obj_id}_{violation_id}_{ip_address}.mp4"
    image_filename = f"{current_time}_{obj_id}_{violation_id}_{ip_address}_full.jpeg"
    video_path = os.path.join(violation_video_path, video_filename)
    image_path = os.path.join(violation_video_path, image_filename)
    # frames_directory = "./frames/"
    # Create directories if they do not exist
    os.makedirs(violation_video_path, exist_ok=True)

    # Initialize video writer for H.264 encoding in HD
    video_size = (1280, 720)  # HD resolution
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), FRAMES_PER_SECOND, video_size)

    # violation_frame_id = tracker['violation_frame'][0]
    start_frame = max(0, violation_frame_id - 40)
    end_frame = violation_frame_id + 40
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_frame_idx = {
            executor.submit(load_frame_from_jpeg, frame_idx, frames_directory, ip_address): frame_idx
            for frame_idx in range(start_frame, end_frame)
        }
        loaded_frames = {}
        for future in concurrent.futures.as_completed(future_to_frame_idx):
            frame_idx = future_to_frame_idx[future]
            frame = future.result()
            if frame is not None:
                loaded_frames[frame_idx] = frame

        boxes = {}
        for frame_idx in sorted(loaded_frames):
            if frame_idx in tracker['frame_indices']:
                bbox_idx = tracker['frame_indices'].index(frame_idx)
                bbox = tracker['bboxes'][bbox_idx]
                boxes[frame_idx] = bbox

        interpolated_boxes = interpolate_bboxes(boxes)
        for frame_idx in sorted(loaded_frames):
            frame = loaded_frames[frame_idx]
            if frame is not None:
                original_frame_size = (frame.shape[1], frame.shape[0])  # Width and height of the original frame
                full_frame = cv2.resize(frame, video_size)  # Resize to target size
                bbox = boxes.get(frame_idx) or interpolated_boxes.get(frame_idx)

                if isinstance(bbox, list) and len(bbox) == 4:
                    # Scale the bbox for the resized frame
                    scaled_bbox = scale_bbox(bbox, original_frame_size, video_size)
                    # Draw the bbox on the full frame
                    cv2.rectangle(full_frame, (scaled_bbox[0], scaled_bbox[1]), (scaled_bbox[2], scaled_bbox[3]), (0, 0, 255), 2)

                    # Define text position (e.g., top left corner of the bbox)
                    text_position = (scaled_bbox[0], scaled_bbox[1] - 10)
                    cv2.putText(full_frame, f"ID: {obj_id}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                video_writer.write(full_frame)
                if frame_idx == violation_frame_id:
                    with open(image_path, 'wb') as file:
                        file.write(engine.encode(full_frame, quality=100))
                        file.close()

        video_writer.release()

        temp_video_path = video_path.replace('.mp4', '_temp.mp4')

        conversion_command = f"ffmpeg -y -i {video_path} -vcodec libx264 {temp_video_path} >/dev/null 2>&1"
        os.system(conversion_command)
        time.sleep(3)
        # Replace the original file with the converted file
        if os.path.exists(temp_video_path):
            os.replace(temp_video_path, video_path)
        print("Debug: Violation media saved successfully")

    if video_writer.isOpened():
        video_writer.release()
    print("Debug: Violation media processing completed")
