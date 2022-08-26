from skimage.metrics import structural_similarity as ssim, \
    peak_signal_noise_ratio as psnr
from os import environ
from sys import argv
import numpy as np
from tonemap import tonemapping

environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv


def relmse(gt, img):
    return np.nanmean((img - gt) ** 2 / (gt ** 2 + .01))


if __name__ == "__main__":
    assert len(argv) == 3
    ref = np.clip(cv.imread(argv[1], cv.IMREAD_UNCHANGED)[:, :, :3], 0, 10)
    test = np.clip(cv.imread(argv[2], cv.IMREAD_UNCHANGED)[:, :, :3], 0, 10)
    print(f"1-SSIM = {1 - ssim(ref, test, multichannel=True)}")
    print(f"PSNR = {psnr(ref, test, data_range=10)}")
    print(f"relMSE = {relmse(ref, test)}")
