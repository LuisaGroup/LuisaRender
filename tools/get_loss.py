import sys
import os

import numpy as np
import cv2 as cv


def l2_loss(rendered: np.ndarray, target: np.ndarray):
    rendered_shape = rendered.shape
    target_shape = target.shape

    assert len(rendered_shape) > 1
    print(rendered_shape, target_shape)
    assert rendered_shape[:-1] == target_shape[:-1]

    pixel_num = 1
    for i in range(len(rendered_shape) - 1):
        pixel_num *= rendered_shape[i]

    return np.square(rendered - target).sum() / pixel_num
    # return (rendered - target).sum() / pixel_num


def read_images_from_dir(dir):
    print(f"Reading images from '{dir}'...")
    image_filenames = sorted(f for f in os.listdir(dir) if f.endswith(".exr"))
    images = [cv.imread(f"{dir}/{f}", cv.IMREAD_UNCHANGED)[:, :, :3] for f in image_filenames]
    return images


if __name__ == '__main__':
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    target_filename = sys.argv[2]
    target_image = cv.imread(target_filename, cv.IMREAD_UNCHANGED)[:, :, :3]
    rendered_dir = sys.argv[1]
    rendered_images = read_images_from_dir(rendered_dir)

    for i in range(1, len(rendered_images)):
        loss = l2_loss(rendered_images[i], target_image)
        print(i, loss)
