import cv2 as cv
from sys import argv
import os

if __name__ == "__main__":
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    folder = argv[1]
    frames = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
    frames = [cv.imread(f"{folder}/{f}", cv.IMREAD_COLOR) for f in frames]
    frame_rate = 10
    video = cv.VideoWriter(f"{folder}/video.mp4", cv.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_rate,
                           frames[0].shape[:2])
    for i, frame in enumerate(frames):
        print(f"Frame {i}")
        video.write(frame)
