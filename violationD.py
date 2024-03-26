from save import save_violation_video_if_needed
import queue
import threading
from os import path as OsPath


save_task_queue = queue.Queue()

farthest_line = []


def save_task(obj_id, frame_id, ip_address, tracker, has_violation, detected_violations, violation_video_path, farthest_line):
    save_violation_video_if_needed(obj_id, frame_id, ip_address, tracker, has_violation, detected_violations, violation_video_path, farthest_line)


# Define the save_worker function
def save_worker():
    while True:
        task = save_task_queue.get()
        if task is None:
            break  # Allows the thread to be shut down
        save_task(*task)
        save_task_queue.task_done()


worker_thread = threading.Thread(target=save_worker)
worker_thread.start()


class ViolationDetector:
    def __init__(self, name: str = 'None') -> None:
        self.__name = name

        self.tracker = {}
        self.violations = {
            'red_light': set(),
            'stop_line': set(),
            '1.1_line': set(),
            'incorrect_direction': set()
        }

        self.status = {}

        self.violation_video_path = OsPath.join(OsPath.dirname(__file__), "detections")
        self.violation_image_path = OsPath.join(OsPath.dirname(__file__), "detections")

        self.last_two_bboxes = {}
        self.detected_violations = {}
        self.stop_line_crossing_frame_id = {}
        self.car_direction = {}
        self.indirect = {}

    def __repr__(self) -> str:
        return f'ViolationDetector(name={self.__name})'

    def __str__(self) -> str:
        return self.__name

    # Part of Done
    def update_tracker(self, frame_id, detected_objects, ip_address):
        global farthest_line
        current_tracked_ids = set(obj['id'] for obj in detected_objects)
        for obj in detected_objects:
            obj_id = obj['id']
            tracker_obj = self.tracker.get(obj_id)

            if tracker_obj is None:
                self.tracker[obj_id] = {
                    **obj,
                    'first_seen': frame_id,
                    'last_seen': frame_id,
                    'frame_indices': [frame_id],
                    'bboxes': [obj['bbox']],
                    'pbboxes': [obj['pbbox']],
                    'missing_frames': 0,
                    'violation_frame': []
                }
            else:
                tracker_obj.update({'last_seen': frame_id, 'missing_frames': 0})
                tracker_obj['frame_indices'].append(frame_id)
                tracker_obj['bboxes'].append(obj['bbox'])
                tracker_obj['pbboxes'].append(obj['pbbox'])
        # Increment missing_frames for objects not currently tracked
        for obj_id, tracker_obj in self.tracker.items():
            if obj_id not in current_tracked_ids:
                tracker_obj['missing_frames'] += 1

        # Identify lost IDs for batch deletion
        lost_ids = {obj_id for obj_id, obj_data in self.tracker.items() if obj_data['missing_frames'] > 20}
        # Batch deletion and task queue addition
        for obj_id in lost_ids:
            task = (
                obj_id,
                frame_id,
                ip_address,
                self.tracker[obj_id],
                self.has_violation(obj_id),
                self.detected_violations,
                self.violation_video_path,
                farthest_line)

            save_task_queue.put(task)
            del self.tracker[obj_id]

            if obj_id in self.car_direction:
                del self.car_direction[obj_id]
            if obj_id in self.stop_line_crossing_frame_id:
                del self.stop_line_crossing_frame_id[obj_id]
            del self.indirect[obj_id]

    # Done
    def update_last_two_bboxes(self, obj_id, bbox):
        if obj_id not in self.last_two_bboxes:
            self.last_two_bboxes[obj_id] = [None, None]
        # Only update if the new bbox is different from the last one
        if self.last_two_bboxes[obj_id][-1] != bbox:
            self.last_two_bboxes[obj_id].pop(0)
            self.last_two_bboxes[obj_id].append(bbox)

    # Done
    def update_status(self, obj_id, new_status, traffic_light_status):
        if obj_id not in self.status:
            self.status[obj_id] = []

        combined_status = (new_status, traffic_light_status)

        # Append the combined tuple to the status list for the obj_id
        self.status[obj_id].append(combined_status)

    # Done
    def check_tl_light_violation(self, obj_id, frame_id, car_bbox, stop_line, red_line, traffic_light_status):
        status_updates = self.status.get(obj_id, [])
        simple_status_updates = [status[0] for status in status_updates]

        if 'stop' not in simple_status_updates:
            if self.is_line_between_bboxes(car_bbox, stop_line, obj_id):
                self.update_status(obj_id, 'stop', traffic_light_status)
                self.stop_line_crossing_frame_id[obj_id] = frame_id

        if 'red' not in simple_status_updates:
            if self.is_line_between_bboxes(car_bbox, red_line, obj_id):
                self.update_status(obj_id, 'red', traffic_light_status)

        status_updates = self.status.get(obj_id, [])
        # Check for 'red_light' violation (exact sequence: [('stop', 'Red'), ('red', 'Red')])
        if status_updates == [('stop', 'Red'), ('red', 'Red')] or status_updates == [('stop', 'Red'), ('red', 'RedYellow')]:
            return 'red_light'
        elif len(status_updates) >= 2 and (status_updates[0] == ('stop', 'Red') or status_updates[0] == ('stop', 'RedYellow')):
            return 'stop_line'
        # Check for 'stop_line' violation (exact sequence: [('stop', 'Red'), ('red', '!Red')])

        return None

    # Done
    def check_1_1_line_violation(self, car_bbox, line_positions):
        if car_bbox and self.touches_line(car_bbox, line_positions['1.1_line']):
            for line in line_positions['1.1_line']:
                if self.touches_line(car_bbox, line):
                    # print("Debug: 1.1 line violation detected.")
                    return True
        return False

    # Done
    def incorrect_direction_violation(self, obj_id, car_bbox, line_positions):
        global farthest_line

        if obj_id not in self.car_direction:
            self.car_direction[obj_id] = []  # Initialize a new list for this obj_id if it doesn't exist
        if obj_id not in self.indirect:
            self.indirect[obj_id] = []

        # Identify the farthest line for comparison
        farthest_line = min(line_positions['1.1_line'], key=lambda line: line[0])
        lx1 = farthest_line[0][0]  # y1 from the first point
        ly1 = farthest_line[0][1]  # y1 from the first point
        lx2 = farthest_line[1][0]  # y1 from the first point
        ly2 = farthest_line[1][1]  # y2 from the second point

        y = min(ly1, ly2)
        if car_bbox[3] > y:
            self.car_direction[obj_id].append(car_bbox)
        if len(self.car_direction[obj_id]) < 5:
            return False

        # Extract first and last bounding boxes
        # carBBOX => [ [10, 10, 20, 20], [10, 10, 20, 20] ]

        first_bbox = self.car_direction[obj_id][0]
        last_bbox = self.car_direction[obj_id][-1]

        # Determine the direction of the car
        if first_bbox[1] < last_bbox[1] and first_bbox[1] * 1.2 < last_bbox[1]:
            direction = "coming closer"
        elif first_bbox[1] > last_bbox[1] and last_bbox[1] * 1.2 < first_bbox[1]:
            direction = "moving away"
        else:
            direction = "Standing"

        m = (ly2 - ly1)/(lx2 - lx1)
        b = ly1 - m * lx1
        for bbox in self.car_direction[obj_id]:
            x = (bbox[3] - b) / m
            if direction == 'coming closer':
                self.indirect[obj_id].append(bbox[0] > x)
            elif direction == 'moving away':
                self.indirect[obj_id].append(bbox[2] < x)
            else:
                self.indirect[obj_id].append(False)

        true_count = sum(self.indirect[obj_id])
        if true_count > 5:
            return True
        else:
            return False

    # Done
    def check_violations(self, frame_id, traffic_light_status, line_positions, detected_objects):
        for obj in detected_objects:
            obj_id = obj['id']
            car_bbox = obj['bbox']
            self.update_last_two_bboxes(obj_id, car_bbox)
            stop_line = line_positions['stop_line'][0]
            red_line = line_positions['red_line'][0]

            # Initial violation check
            if 'red_light' not in self.detected_violations.get(obj_id, set()):
                light_violation = self.check_tl_light_violation(obj_id, frame_id, car_bbox, stop_line, red_line, traffic_light_status)
                if light_violation == 'red_light':
                    # If red light violation is detected, record and skip other checks
                    self.record_violation(obj_id, 'red_light', frame_id)
                    continue
                elif light_violation == 'stop_line':
                    violation_frame_id = self.stop_line_crossing_frame_id.get(obj_id, frame_id)
                    self.record_violation(obj_id, 'stop_line', violation_frame_id)
                    continue

            if '1.1_line' not in self.detected_violations.get(obj_id, set()):
                # Check 1.1 line violation
                if self.check_1_1_line_violation(car_bbox, line_positions):
                    self.record_violation(obj_id, '1.1_line', frame_id)

            if 'incorrect_direction' not in self.detected_violations.get(obj_id, set()):
                if self.incorrect_direction_violation(obj_id, car_bbox, line_positions):
                    self.record_violation(obj_id, 'incorrect_direction', frame_id)

    # Done
    def record_violation(self, obj_id, violation_type, frame_id):
        # Record the violation if not already detected
        if violation_type not in self.detected_violations.get(obj_id, set()):
            self.violations[violation_type].add(obj_id)
            self.detected_violations.setdefault(obj_id, set()).add(violation_type)
            self.tracker[obj_id]['violation_frame'].append((violation_type, frame_id))

    # Done
    def has_violation(self, obj_id):
        return any(obj_id in self.violations[violation] for violation in self.violations)

    # Done
    def touches_line(self, bbox, line_segments):
        # Ensure line_segments is a list
        if not isinstance(line_segments, list):
            line_segments = [line_segments]

        modified_bbox = list(bbox)

        width = (modified_bbox[2] - modified_bbox[0])
        shrink = (width * 0.15)
        modified_bbox[0] = round(modified_bbox[0] + shrink)
        modified_bbox[2] = round(modified_bbox[2] - shrink)

        bbox_edges = [
            (modified_bbox[0], bbox[3], modified_bbox[2], bbox[3]),  # Bottom edge of bbox
        ]
        for line_segment in line_segments:  # Iterate over each line segment
            line_start, line_end = line_segment
            for edge in bbox_edges:
                if self.do_lines_intersect(edge, line_start + line_end):
                    return True
        return False

    # Done
    def is_line_between_bboxes(self, current_bbox, line, obj_id):

        if obj_id not in self.last_two_bboxes or len(self.last_two_bboxes[obj_id]) != 2:
            return False

        last_bbox = self.last_two_bboxes[obj_id][0]
        current_bottom_line = (current_bbox[0], current_bbox[3], current_bbox[2], current_bbox[3])
        width = (current_bbox[3] - current_bbox[1])
        cshrink = (width * 0.25)

        if last_bbox:
            width = (last_bbox[3] - last_bbox[1])
            lshrink = (width * 0.25)
            last_bottom_line = (last_bbox[0], last_bbox[3], last_bbox[2], last_bbox[3])
            # print (last_bottom_line, current_bottom_line)
            # Check if the line is between these two bottom lines
            return self.does_line_intersect_or_between(line, current_bottom_line, last_bottom_line, cshrink, lshrink)
        else:
            # If no last bounding box, just check with the current bounding box
            return self.does_line_intersect_or_between(line, current_bottom_line, current_bottom_line, cshrink, cshrink)

    # Done
    def does_line_intersect_or_between(self, line, bottom_line1, bottom_line2, cshrink, lshrink):
        line_y1, line_y2 = line[0][1], line[1][1]

        bbox_y1 = bottom_line1[1]  # As bottom_line1 is horizontal, y1 = y2
        bbox_y1 = round(bbox_y1 - cshrink)
        bbox_y2 = bottom_line2[1]
        bbox_y2 = round(bbox_y2 - lshrink)
        top_y = min(bbox_y1, bbox_y2)
        bottom_y = max(bbox_y1, bbox_y2)
        # print (f"line: {line_y1, line_y2}")
        # print (f"plate: {top_y, bottom_y}")
        if (line_y1 >= top_y and line_y1 <= bottom_y) or (line_y2 >= top_y and line_y2 <= bottom_y) or (line_y1 >= top_y and line_y2 <= bottom_y) or (line_y2 >= top_y and line_y1 <= bottom_y):
            return True
        else:
            return False

    # Done
    def do_lines_intersect(self, line1, line2):
        """
        Check if two line segments intersect.
        Each line is defined by four coordinates (x1, y1, x2, y2).
        """
        # Unpack points
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Calculate determinants
        det1 = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        det2 = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        det3 = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)

        # Check if lines are parallel
        if det1 == 0:
            return False  # Lines are parallel

        # Calculate intersection point
        t = det2 / det1
        u = -det3 / det1

        # Check if intersection is within line segments
        return 0 <= t <= 1 and 0 <= u <= 1
