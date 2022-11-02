import shutil
from pathlib2 import Path
from sys import argv
import glm
import json


def rotateXYZ(R):
    return glm.rotate(R.z, (0, 0, 1)) * glm.rotate(R.y, (0, 1, 0)) * glm.rotate(R.x, (1, 0, 0))


def rotateYXZ(R):
    return glm.rotate(R.z, (0, 0, 1)) * glm.rotate(R.x, (1, 0, 0)) * glm.rotate(R.y, (0, 1, 0))


def convert_camera(camera: dict) -> dict:
    resolution_ratio = camera["ratio"]
    name = camera["name"]
    fov = camera["fov"]
    position = glm.vec3(camera["eye"])
    look_at = glm.vec3(camera["look"])
    front = glm.normalize(look_at - position)
    up = glm.vec3(camera["up"])

    camera_dict = {
        name: {
            "type": "Camera",
            "impl": "Pinhole",
            "prop": {
                "position": list(position),
                "front": list(front),
                "up": list(up),
            },
            "fov": fov,
            "spp": 1,
            "film": {
                "type": "Film",
                "impl": "Color",
                "prop": {
                    "resolution": [
                        1024,
                        int(1024 / resolution_ratio)
                    ],
                    "exposure": 0,
                }
            },
            "file": "output.exr",
            "filter": {
                "impl": "Gaussian",
                "prop": {
                    "radius": 1
                }
            }
        }
    }
    return camera_dict


def check_path(input_path: Path, output_path: Path):
    if not Path.exists(input_path):
        raise FileNotFoundError(f"Path {input_path} not found")
    if not Path.exists(output_path):
        Path.mkdir(output_path)


def disney2luisa(input_path: Path, output_path: Path):
    shutil.rmtree(output_path)
    check_path(input_path, output_path)

    path_in = input_path / "json"
    path_out = output_path / "json"
    check_path(path_in, path_out)

    path_in = path_in / "cameras"
    path_out = path_out / "cameras"
    check_path(path_in, path_out)
    for camera_file in path_in.iterdir():
        with open(camera_file, "r") as f:
            camera = json.load(f)
        camera_dict = convert_camera(camera)
        with open(path_out / camera_file.name, "w") as f:
            json.dump(camera_dict, f, indent=2)


if __name__ == '__main__':
    if len(argv) == 3:
        disney2luisa(Path(argv[1]), Path(argv[2]))
    else:
        print('Usage: disney2luisa.py <input.json> <output.luisa>')