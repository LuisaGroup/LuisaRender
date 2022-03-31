import cv2 as cv
from sys import argv
import numpy as np

if __name__ == "__main__":
    file_name = argv[1]
    size = int(argv[2])
    image = 127 * np.ones(shape=[size, size, 3])
    cv.imwrite(file_name, np.uint8(image))
