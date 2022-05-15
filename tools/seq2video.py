from sys import argv
from os import listdir, environ
import numpy as np


def rgb2srgb(image):
    return np.uint8(np.round(np.clip(np.where(
        image <= 0.00304,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055
    ) * 255, 0, 255)))


def exr2png(images):
    for i, image in enumerate(images):
        print(f"Transforming frame {i}")
        frame = rgb2srgb(image)
        cv.imwrite(f"{folder}/{files[i][:-4]}.png", frame)


def png2video(images, frame_rate):
    writer = cv.VideoWriter(f"{folder}/output.mp4", cv.VideoWriter_fourcc('m', 'p', '4', 'v'),
                            frame_rate, (images[0].shape[1], images[0].shape[0]))
    # writer = cv.VideoWriter(f"{folder}/output.avi", cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
    #                         frame_rate, (images[0].shape[1], images[0].shape[0]))
    for i, image in enumerate(images):
        print(f"Processing frame {i}")
        writer.write(image)

    writer.release()
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":
    environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2 as cv

    folder = argv[1]
    frame_rate = int(argv[2])

    print(f"Reading images from '{folder}'...")
    files = sorted(f for f in listdir(folder) if f.endswith(".exr") and not f.startswith("dump-"))
    images = [cv.imread(f"{folder}/{f}", cv.IMREAD_UNCHANGED) for f in files]
    exr2png(images)

    print("")

    files = sorted(f for f in listdir(folder) if f.endswith(".png") and not f.startswith("dump-"))
    print(f"Reading images from '{folder}'...")
    images = [cv.imread(f"{folder}/{f}", cv.IMREAD_COLOR) for f in files]
    png2video(images, frame_rate)
