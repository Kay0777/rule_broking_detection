import cv2
# import numpy as np
global current_line_type
current_line_type = ""
global ix, iy
ix, iy = -1, -1

# Define colors for different types of lines
colors = {
    "1.1_line": (255, 0, 0),  # Blue
    "stop_line": (0, 255, 0),  # Green
    "red_line": (0, 0, 255)  # Red
}

# Initialize a dictionary to store line information for each camera
lines_info = {}
original_frame_sizes = {}
# Mouse callback function for drawing lines on the image


def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, img, current_lines_info, current_line_type

    # Only start drawing if a label is selected
    if current_line_type == "":
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a line
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Draw a line on a copy of the image to show real-time drawing
        img_copy = img.copy()
        cv2.line(img_copy, (ix, iy), (x, y), colors[current_line_type], 2)
        cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing the line
        drawing = False
        cv2.line(img, (ix, iy), (x, y), colors[current_line_type], 2)
        current_lines_info.append(((ix, iy), (x, y), current_line_type))
        cv2.imshow('image', img)


def select_label(key):
    rules = {
        49: "1.1_line",  # key => 1
        50: "stop_line",  # key => 2
        51: "red_line"  # key => 3
    }
    return rules.get(key, "Invalid key. Please select the label correctly.")

# Function to display the first frame and let the user draw lines


def get_first_frame_and_draw_lines(cap, stream_id):
    global img, drawing, current_lines_info, current_line_type
    drawing = False
    current_lines_info = []

    ret, frame = cap.read()
    if not ret:
        print(f"Failed to grab the first frame from stream {stream_id}.")
        return []

    original_frame_sizes[stream_id] = frame.shape[1::-1]

    # Resize the image for drawing purposes
    display_size = (640, 640)
    img = cv2.resize(frame, display_size)

    # Calculate scaling factors
    scale_x = frame.shape[1] / display_size[0]
    scale_y = frame.shape[0] / display_size[1]

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_line)

    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # Exit drawing mode
            break
        elif k in [ord('1'), ord('2'), ord('3'), ord('4')]:  # Label selection
            current_line_type = select_label(k)

    cv2.destroyAllWindows()

    # Adjust line coordinates to the original frame size
    adjusted_lines_info = []
    for line_info in current_lines_info:
        start_point, end_point, line_type = line_info  # Assuming line_info is a tuple of ((start_x, start_y), (end_x, end_y), line_type)
        adjusted_start = (int(start_point[0] * scale_x), int(start_point[1] * scale_y))
        adjusted_end = (int(end_point[0] * scale_x), int(end_point[1] * scale_y))
        adjusted_lines_info.append((adjusted_start, adjusted_end, line_type))

    return adjusted_lines_info

# Main function to process each stream


def process_streams(rtsp_urls):
    for stream_id, url in enumerate(rtsp_urls, start=0):
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"Failed to open stream {stream_id}.")
            continue

        print(f"Processing stream {stream_id}. Press 'q' to finish drawing lines.")
        lines_info[stream_id] = get_first_frame_and_draw_lines(cap, stream_id)
        cap.release()

        # Display drawn lines info
        print(f"Drawn Lines Information for stream {stream_id}:")
        for line in lines_info[stream_id]:
            print(line)

    return lines_info
