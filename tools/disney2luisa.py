import shutil
from pathlib2 import Path
from sys import argv
import glm
import json


def convert_camera(camera: dict) -> (str, dict):
    position = glm.vec3(camera['eye'])
    look_at = glm.vec3(camera['look'])
    front = glm.normalize(look_at - position)
    up = glm.vec3(camera['up'])

    camera_dict = {
        'type': 'Camera',
        'impl': 'Pinhole',
        'prop': {
            'position': list(position),
            'front': list(front),
            'up': list(up),
            'fov': camera['fov'],
            'spp': 1,
            'film': {
                'type': 'Film',
                'impl': 'Color',
                'prop': {
                    'resolution': [
                        1024,
                        int(1024 / camera['ratio'])
                    ],
                    'exposure': 0,
                }
            },
            'file': 'output.exr',
            'filter': {
                'type': 'Filter',
                'impl': 'Gaussian',
                'prop': {
                    'radius': 1
                }
            }
        }
    }

    return camera['name'], camera_dict


def convert_light(light: dict, name: str) -> (str, dict):
    transform = light['translationMatrix']
    exposure = light['exposure']
    color = glm.vec4(light['color']).xyz * exposure
    light_dict = {}

    if light['type'] == 'quad':
        light_dict = {
            'type': 'Shape',
            'impl': 'Mesh',
            'prop': {
                'file': 'quad.obj',
                'surface': {
                    'type': 'Surface',
                    'impl': 'Null',
                },
                'transform': {
                    'type': 'Transform',
                    'impl': 'Stack',
                    'prop': {
                        'transforms': [
                            {
                                'type': 'Transform',
                                'impl': 'SRT',
                                'prop': {
                                    'scale': [light['width'] / 2, light['height'] / 2, 1],
                                }
                            },
                            {
                                'type': 'Transform',
                                'impl': 'Matrix',
                                'prop': {
                                    'm': transform,
                                }
                            },
                        ]
                    }
                },
                'light': {
                    'type': 'Light',
                    'impl': 'Diffuse',
                    'prop': {
                        'emission': {
                            'type': 'Texture',
                            'impl': 'Constant',
                            'prop': {
                                'v': list(color),
                            }
                        }
                    }
                }
            }
        }
    elif light['type'] == 'dome':
        light_dict = {
            'type': 'Environment',
            'impl': 'Spherical',
            'prop': {
                'emission': {
                    'type': 'Texture',
                    'impl': 'Image',
                    'prop': {
                        'file': light['envmapCamera'],
                    }
                },
                'transform': {
                    'type': 'Transform',
                    'impl': 'Stack',
                    'prop': {
                        'transforms': [
                            {
                                'type': 'Transform',
                                'impl': 'SRT',
                                'prop': {
                                    'scale': [-1, 1, 1],
                                    'rotate': [-1, 0, 0, 90],
                                }
                            },
                            {
                                'type': 'Transform',
                                'impl': 'SRT',
                                'prop': {
                                    'rotate': [0, 0, 1, 65],
                                }
                            },
                        ]
                    }
                },
            }
        }
    else:
        print(f'Light type {light["type"]} not supported')
        return None, None

    return name, light_dict


def check_path(input_path: Path, output_path: Path):
    if not Path.exists(input_path):
        raise FileNotFoundError(f'Path {input_path} not found')
    if not Path.exists(output_path):
        Path.mkdir(output_path)


def disney2luisa(input_path: Path, output_path: Path):
    shutil.rmtree(output_path)
    check_path(input_path, output_path)

    path_in = input_path / 'json'
    path_out = output_path / 'json'
    check_path(path_in, path_out)

    # cameras
    path_in = path_in / 'cameras'
    path_out = path_out / 'cameras'
    check_path(path_in, path_out)
    for camera_file in path_in.iterdir():
        with open(camera_file, 'r') as f:
            camera = json.load(f)
        name, camera_dict = convert_camera(camera)
        camera_dict = {name: camera_dict}
        with open(path_out / camera_file.name, 'w') as f:
            json.dump(camera_dict, f, indent=2)
    path_in = path_in.parent
    path_out = path_out.parent

    # lights
    path_in = path_in / 'lights'
    path_out = path_out / 'lights'
    check_path(path_in, path_out)
    quad = '''
v   1.00   1.00   0.00
v  -1.00   1.00   0.00
v  -1.00  -1.00   0.00
v   1.00  -1.00   0.00
f -4 -3 -2 -1
'''
    with open(path_out / 'quad.obj', 'w') as f:
        f.write(quad)
    for light_file in path_in.iterdir():
        lights_dict = {}
        with open(light_file, 'r') as f:
            lights = json.load(f)
        for name, light in lights.items():
            name, light_dict = convert_light(light, name)
            if name is None:
                continue
            light_dict = {name: light_dict}
            lights_dict.update(light_dict)
        with open(path_out / light_file.name, 'w') as f:
            json.dump(lights_dict, f, indent=2)


if __name__ == '__main__':
    if len(argv) == 3:
        disney2luisa(Path(argv[1]), Path(argv[2]))
    else:
        print('Usage: disney2luisa.py <input.json> <output.luisa>')
