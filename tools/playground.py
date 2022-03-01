import cv2 as cv


if __name__ == "__main__":
    image = cv.imread(r"C:\Users\Mike\Desktop\LuisaRender\data\scenes\living-room\textures\leaf.png", cv.IMREAD_UNCHANGED)
    cv.imwrite(r"C:\Users\Mike\Desktop\LuisaRender\data\scenes\living-room\textures\leaf-alpha.png", image[..., -1])
