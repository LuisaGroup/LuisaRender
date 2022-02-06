import cv2 as cv


if __name__ == "__main__":
    image = cv.imread("/Users/mike/Downloads/living-room/textures/leaf.png", cv.IMREAD_UNCHANGED)
    cv.imwrite("/Users/mike/Downloads/living-room/textures/leaf-alpha.png", image[::-1, :, -1])
