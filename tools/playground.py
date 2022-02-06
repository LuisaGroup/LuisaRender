import cv2 as cv


if __name__ == "__main__":
    image = cv.imread("/Users/mike/Downloads/bathroom/textures/rug_mask.png", cv.IMREAD_UNCHANGED)
    cv.imwrite("/Users/mike/Downloads/bathroom/textures/rug_mask-alpha.png", image[::-1, :, -1])
