import cv2
from dependency import Frame

async def capture_frame_process_1(ip: str, url: str) -> None:
    frame: Frame = Frame(ip=ip)
    
    while 1:
        cap: cv2.VideoCapture = cv2.VideoCapture(url)
        while 1:
            ret, frame = cap.read()
            if not ret:
                frame.set_last_frame(frame=None)
                continue
            
            frame.set_last_frame(frame=frame)

        print(f'{ret=}, {type(ret)=}')
        print(type(frame))
    
async def saving_frame_process_2(ip: str) -> None:
    frame: Frame = Frame(ip=ip)
    await frame.save_image(ip=ip)

    

# frame_save_queue = queue.Queue()
# async def frame_capture_loop(rtsp_url: str) -> None:
#     async with Camera(url=rtsp_url) as cam:
#         ...
    
    
#     while 1:  # Outer loop to handle reconnection
#         try:
#             cap = cv2.VideoCapture(rtsp_url)
#             frame_id = 0
#             while 1:
#                 ret, frame = cap.read()
#                 FRAME = Frame(frame=frame)
#                 if ret:
#                     with threading.Lock():
#                         if len(global_video_buffer) >= MAX_FRAMES_IN_BUFFER:
#                             oldest_frame_id = next(iter(global_video_buffer))
#                             del global_video_buffer[oldest_frame_id]
#                         global_video_buffer[frame_id] = frame
#                     frame_save_queue.put((frame_id, frame))
#                     frame_id += 1
#                 else:
#                     print("Lost connection to the camera. Attempting to reconnect...")
#                     cap.release()  # Release the current connection
#                     break  # Exit the inner loop to attempt reconnection

#             # Optional: Add a delay before trying to reconnect
#             time.sleep(5)  # Wait for 5 seconds before trying to reconnect

#         finally:
#             cap.release()