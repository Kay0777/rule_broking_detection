import numpy as np
import cv2
import os


def main():
    folder = 'frames/10.42.6.72'
    files = [os.path.join(folder, file) for file in sorted(os.listdir(folder), key=lambda file: int(file.split('.')[0]))]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(
        filename='a.mp4',
        fourcc=fourcc,
        fps=10,
        frameSize=(1280, 720))

    for file in files[:80]:
        data = np.load(file=file)
        image = cv2.resize(src=data, dsize=(1280, 720))
        videoWriter.write(image=image)

        # cv2.imshow('Kai', data)
        # if cv2.waitKey(0) * 0XFF == ord('c'):
        #     continue

    if videoWriter.isOpened():
        videoWriter.release()


if __name__ == '__main__':
    main()
