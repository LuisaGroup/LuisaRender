import sys
import os

import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv


def l2_loss(rendered: np.ndarray, target: np.ndarray) -> float:
    rendered_shape = rendered.shape
    target_shape = target.shape

    assert len(rendered_shape) > 1
    assert rendered_shape[:-1] == target_shape[:-1]

    pixel_num = 1
    for i in range(len(rendered_shape) - 1):
        pixel_num *= rendered_shape[i]

    return np.square(rendered - target).sum() / pixel_num


def read_hdr2ldr(filename: str) -> np.ndarray:
    image = np.maximum(
        np.nan_to_num(cv.imread(filename, cv.IMREAD_UNCHANGED)[:, :, :3], nan=0.0, posinf=1e3, neginf=0), 0.0)
    return np.uint8(np.round(np.clip(np.where(
        image <= 0.00304,
        12.92 * image,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055
    ), 0.0, 1.0)))


def read_images_from_dir(dir: str) -> list:
    print(f"Reading images from '{dir}'...")
    image_filenames = sorted(f for f in os.listdir(dir) if f.endswith(".exr") and not f.endswith('ref.exr'))
    images = [read_hdr2ldr(f"{dir}/{f}") for f in image_filenames]
    return images


if __name__ == '__main__':
    target_filename = sys.argv[2]
    target_image = read_hdr2ldr(target_filename)
    rendered_dir = sys.argv[1]
    rendered_images = read_images_from_dir(rendered_dir)

    for i in range(len(rendered_images)):
        loss = l2_loss(rendered_images[i], target_image)
        print(i, loss)
