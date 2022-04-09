from os import environ
from sys import argv


if __name__ == "__main__":
    environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2 as cv
    if len(argv) < 2:
        print("Usage: rgba2rgb.py <image>")
        exit(1)
    image = cv.imread(argv[1], cv.IMREAD_UNCHANGED)
    assert image.shape[2] == 4
    ext = f'.{argv[1].split(".")[-1]}'
    cv.imwrite(f'{argv[1][:-len(ext)]}-rgb{ext}', image[:, :, :3])
