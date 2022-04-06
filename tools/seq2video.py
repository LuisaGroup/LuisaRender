from sys import argv
from os import listdir, environ
import numpy as np


def rgb2srgb(image):
    return np.uint8(np.round(np.clip(np.where(
        image <= 0.00304,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055
    ) * 255, 0, 255)))


if __name__ == "__main__":
    environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2 as cv

    folder = argv[1]
    files = sorted(f for f in listdir(folder) if f.endswith(".exr"))
    print(f"Reading images from '{folder}'...")
    images = [cv.imread(f"{folder}/{f}", cv.IMREAD_UNCHANGED) for f in files]
    writer = cv.VideoWriter(f"{folder}/output.avi", cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, images[0].shape[:2])
    for i, image in enumerate(images):
        print(f"Processing frame {i}")
        frame = rgb2srgb(image)
        cv.imshow("Display", frame)
        cv.waitKey(1)
        cv.imwrite(f"{folder}/{files[i][:-4]}.png", frame)
        writer.write(frame)
    writer.release()
    cv.waitKey()
