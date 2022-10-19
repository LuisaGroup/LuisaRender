from os import environ
from sys import argv


if __name__ == "__main__":
    environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2 as cv
    if len(argv) < 2:
        print("Usage: rgba2trans.py <image>")
        exit(1)
    image = cv.imread(argv[1], cv.IMREAD_UNCHANGED)
    assert image.shape[2] == 4
    ext = f'.{argv[1].split(".")[-1]}'.lower()
    alpha = image[:, :, -1]
    cv.imwrite(f'{argv[1][:-len(ext)]}-trans{ext}', 1 - alpha if ext == ".exr" else 255 - alpha)
