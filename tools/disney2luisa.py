import json
from pathlib import Path
from sys import argv
import glm


def rotateXYZ(R):
    return glm.rotate(R.z, (0, 0, 1)) * glm.rotate(R.y, (0, 1, 0)) * glm.rotate(R.x, (1, 0, 0))


def rotateYXZ(R):
    return glm.rotate(R.z, (0, 0, 1)) * glm.rotate(R.x, (1, 0, 0)) * glm.rotate(R.y, (0, 1, 0))


def disney2luisa(filename: str, output_filename: str):
    pass


if __name__ == '__main__':
    if len(argv) == 3:
        disney2luisa(argv[1], argv[2])
    else:
        print('Usage: disney2luisa.py <input.json> <output.luisa>')