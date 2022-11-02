import shutil
from pathlib2 import Path
from sys import argv
import glm
import json
import numpy as np
import re

material_map = {}
re_material_map = {}


def check_path(input_path: Path, output_path: Path):
    if not Path.exists(input_path):
        raise FileNotFoundError(f'Path {input_path} not found')
    if not Path.exists(output_path):
        Path.mkdir(output_path)


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
    transform = np.array(light['translationMatrix'], dtype=np.float32)
    transform = transform.reshape(4, 4).transpose()
    exposure = light['exposure']
    color = glm.vec4(light['color']).xyz * exposure

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
                                    'm': transform.reshape(16).tolist(),
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


def convert_material(material: dict, name: str) -> (str, dict):
    if name == 'hidden':
        material_dict = {
            'type': 'Surface',
            'impl': 'Null',
        }
    else:
        thin = material['type'] == 'thin'
        prop = {
            'color': material['baseColor'],
            'thin': thin,
            'metallic': material['metallic'],
            'eta': material['ior'],
            'roughness': material['roughness'],
            'specular_tint': material['specularTint'],
            'anisotropic': material['anisotropic'],
            'sheen': material['sheen'],
            'sheen_tint': material['sheenTint'],
            'clearcoat': material['clearcoat'],
            'clearcoat_gloss': material['clearcoatGloss'],
            'specular_trans': material['specTrans'],
        }
        if thin:
            prop['flatness'] = material['flatness']
            prop['diffuse_trans'] = material['diffTrans']

        material_dict = {
            'type': 'Surface',
            'impl': 'Disney',
            'prop': prop
        }

    assignment = material['assignment']
    for shape in assignment:
        if '*' in shape:
            shape = shape.replace('*', '[0-9a-zA-Z_]*')
            re_material_map[shape] = name
        else:
            material_map[shape] = name

    return name, material_dict


def disney2luisa(input_path: Path, output_path: Path):
    shutil.rmtree(output_path)
    check_path(input_path, output_path)

    path_in = input_path / 'json'
    path_out = output_path / 'json'
    check_path(path_in, path_out)
    geo_names = [x.name for x in path_in.iterdir()
                 if x.is_dir() and x.name != 'cameras' and x.name != 'lights']

    # cameras
    path_in = input_path / 'json' / 'cameras'
    path_out = output_path / 'json' / 'cameras'
    check_path(path_in, path_out)
    for camera_file in path_in.iterdir():
        with open(camera_file, 'r') as f:
            camera = json.load(f)
        name, camera_dict = convert_camera(camera)
        camera_dict = {name: camera_dict}
        with open(path_out / camera_file.name, 'w') as f:
            json.dump(camera_dict, f, indent=2)

    # lights
    path_in = input_path / 'json' / 'lights'
    path_out = output_path / 'json' / 'lights'
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

    # geometries
    for geo_name in geo_names:
        path_in = input_path / 'json' / geo_name
        path_out = output_path / 'json' / geo_name
        check_path(path_in, path_out)

        # materials
        material_file = path_in / 'materials.json'
        with open(material_file, 'r') as f:
            materials = json.load(f)
        materials_dict = {}
        for name, material in materials.items():
            name, material_dict = convert_material(material, name)
            material_dict = {name: material_dict}
            materials_dict.update(material_dict)
        with open(path_out / 'materials.json', 'w') as f:
            json.dump(materials_dict, f, indent=2)


if __name__ == '__main__':
    if len(argv) == 3:
        disney2luisa(Path(argv[1]), Path(argv[2]))
    else:
        print('Usage: disney2luisa.py <input.json> <output.luisa>')
