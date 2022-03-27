from sys import argv
import cv2 as cv
import numpy as np


if __name__ == "__main__":
    disp = (cv.imread(argv[1], cv.IMREAD_GRAYSCALE) / 127.5 - 1) / 255
    vertices = np.meshgrid(np.arange(1024))
    print(vertices)
