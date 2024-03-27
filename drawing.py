import numpy as np
import json
import cv2
import os

CAR_COORS = []
PLATE_COORS = []

IMG = None
IS_CAR = True
IS_DRAWING = False

RH, RW, RCH = (2688, 1520, 3)
# VIDEO_SHAPE = (1280, 720)
VIDEO_SHAPE = (RH // 2, RW // 2)

COUNTER = 0

P1 = None
P2 = None


def drawing(event, x, y, flags, param):
    global CAR_COORS, PLATE_COORS, IMG, IS_DRAWING, COUNTER, P1, P2, INDEX

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a line
        IS_DRAWING = True
        P1 = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and IS_DRAWING:
        # Draw a line on a copy of the image to show real-time drawing
        img_copy = IMG.copy()
        color = (0, 0, 255) if COUNTER % 2 else (255, 0, 0)

        cv2.rectangle(
            img=img_copy,
            pt1=P1,
            pt2=(x, y),
            color=color,
            thickness=2,
            lineType=2)
        cv2.imshow(winname='kai', mat=img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing the line
        IS_DRAWING = False
        P2 = (x, y)
        color = (0, 0, 255) if COUNTER % 2 else (255, 0, 0)

        cv2.rectangle(
            img=IMG,
            pt1=P1,
            pt2=P2,
            color=color,
            thickness=2,
            lineType=2)

        cv2.imshow(
            winname='kai',
            mat=IMG)

        if COUNTER % 2:
            coors = [
                [2 * i for i in P1],
                [2 * i for i in P2]
            ]
            CAR_COORS.append({"index": INDEX, "coors": coors})
        else:
            coors = [
                [2 * i for i in P1],
                [2 * i for i in P2]
            ]
            PLATE_COORS.append({"index": INDEX, "coors": coors})
        COUNTER += 1


def write_coor_on_cv2(index: int, data: str) -> None:
    global IMG, IS_CAR, INDEX

    data = cv2.resize(src=data, dsize=VIDEO_SHAPE)
    IMG = data
    INDEX = index

    cv2.namedWindow(winname='kai')
    cv2.imshow('kai', data)
    while True:
        cv2.setMouseCallback(window_name='kai', on_mouse=drawing)
        key: int = cv2.waitKey(delay=1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


def write_file(file, data):
    with open(file, 'w') as f:
        json.dump(obj=data, fp=f, indent=2)
        f.close()


def read_file(file):
    with open(file, 'r') as f:
        data = json.loads(f.read())
        f.close()
    return data


def main():
    global CAR_COORS, PLATE_COORS

    folder = 'frames/10.42.7.72'
    files = [os.path.join(folder, file) for file in sorted(os.listdir(folder), key=lambda file: int(file.split('.')[0]))]
    files = filter(lambda file: os.path.basename(file).split('.')[0] in ['314', '326', '372', '380'], files)

    # for file in files[300:330]:
    for file in files:
        index = int(os.path.basename(file).split('.')[0])
        print(os.path.basename(file))
        data = np.load(file=file)
        data = cv2.resize(src=data, dsize=VIDEO_SHAPE)
        write_coor_on_cv2(index=index, data=data)

    write_file('car.json', CAR_COORS)
    write_file('plate.json', PLATE_COORS)


def main2():
    folder = os.path.join('frames', '10.42.7.72')
    files = [os.path.join(folder, file) for file in sorted(os.listdir(folder), key=lambda file: int(file.split('.')[0]))]

    cars = read_file('car.json')
    plates = read_file('plate.json')

    for file in files[330:332]:
        index = 0
        curIndex = int(file.split('/')[-1].split('.')[0])
        data = np.load(file=file)

        for i, car in enumerate(cars):
            if car['index'] == curIndex:
                index = i

        if index != 0:
            car_coors = cars[index]['coors']
            plate_coors = plates[index]['coors']
            cv2.rectangle(img=data, pt1=tuple(car_coors[0]), pt2=tuple(car_coors[1]), color=(0, 0, 255), thickness=2, lineType=2)
            cv2.rectangle(img=data, pt1=tuple(plate_coors[0]), pt2=tuple(plate_coors[1]), color=(0, 0, 255), thickness=2, lineType=2)
        cv2.imshow('Kai', data)
        if cv2.waitKey() & 0XFF == ord('c'):
            continue


def main3():
    cars = read_file('car.json')
    plates = read_file('plate.json')

    folder = 'frames/10.42.7.72'
    files = [os.path.join(folder, file) for file in sorted(os.listdir(folder), key=lambda file: int(file.split('.')[0]))]
    files = filter(lambda file: os.path.basename(file).split('.')[0] in ['314', '326', '372', '380'], files)

    for file in files:
        index = os.path.basename(file).split('.')[0]
        for info in cars + plates:
            if int(info['index']) == int(index):
                coor = info['coors']
                break

        data = np.load(file)
        # data = cv2.resize(src=data, dsize=VIDEO_SHAPE)
        # cv2.rectangle(img=data, pt1=tuple([i // 2 for i in coor[0]]), pt2=tuple([i // 2 for i in coor[1]]), color=(255, 0, 0))
        # cv2.imshow('Kai', data)
        # if cv2.waitKey() & 0XFF == ord('c'): continue


if __name__ == "__main__":
    # main()
    # main2()
    main3()
