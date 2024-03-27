import numpy as np
import cv2
import os


def main():
    folder = os.path.join('frames', '10.42.7.72')
    shape = ()

    files = [os.path.join(folder, file) for file in sorted(os.listdir(folder), key=lambda file: int(file.split('.')[0]))]

    for file in files:
        data = np.load(file=file)
        shape = tuple(int(i // 2) for i in data.shape[:-1][::-1])
        data = cv2.resize(src=data, dsize=shape)

        print(file)

        cv2.imshow('Kai', data)
        if cv2.waitKey() & 0XFF == ord('c'):
            continue
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
