import shutil
from pathlib2 import Path
from sys import argv
import os
import json
import numpy as np
import re

material_map = {}

re_shape2material = {}

material_names = set()
light_names = set()
camera_names = set()
shape_names = set()


def split_obj(path: Path, output_dir: Path) -> dict:

    class GeometryIndex:
        def __init__(self):
            self.v_index = 0
            self.vt_index = 0
            self.vn_index = 0

        def clear(self):
            self.v_index = 0
            self.vt_index = 0
            self.vn_index = 0

        def __getitem__(self, item: int):
            if item == 0:
                return self.v_index
            elif item == 1:
                return self.vt_index
            elif item == 2:
                return self.vn_index
            else:
                raise IndexError

        def __str__(self):
            return f"v:{self.v_index}, vt:{self.vt_index}, vn:{self.vn_index}"

        def clone(self):
            v = GeometryIndex()
            v.v_index = self.v_index
            v.vt_index = self.vt_index
            v.vn_index = self.vn_index
            return v

    index = GeometryIndex()
    index_next = GeometryIndex()
    geo2material = {}

    with open(path, 'r') as f:
        geo_name = 'default'
        text = ''

        def reindex(line: str, index: GeometryIndex) -> str:
            line = line[2:].strip().split(' ')
            for i in range(len(line)):
                line[i] = line[i].split('/')
                assert len(line[i]) == 3
                for j in range(3):
                    a = line[i][j]
                    if a == '':
                        continue
                    a = int(a)
                    if a >= 0:
                        line[i][j] = str(a - index[j])
                line[i] = '/'.join(line[i])
            return ' '.join(line)

        def write_geo(objtext: str, name: str):
            if name == 'default' or objtext == '':
                return
            new_obj_file_path = output_dir / (name + '.obj')
            if new_obj_file_path.exists():
                raise FileExistsError(f'File {new_obj_file_path} already exists')
            with open(new_obj_file_path, 'w') as f:
                f.write(objtext)

        g_mode = False
        for line in f:
            if line.startswith('g '):
                if g_mode:
                    g_mode = False
                    write_geo(text, geo_name)
                    text = ''
                    index = index_next.clone()

                geo_name = line[2:].strip()
                if geo_name == 'default':
                    continue
                geo_name = f'{path.stem}_{geo_name}'
                assert geo_name not in geo2material
                g_mode = True
            elif line.startswith('f '):
                text += f'f {reindex(line, index)}\n'
            elif line.startswith('usemtl '):
                material = line[7:].strip()
                geo2material[geo_name] = material
            elif line.startswith('v '):
                index_next.v_index += 1
                text += line
            elif line.startswith('vn '):
                index_next.vn_index += 1
                text += line
            elif line.startswith('vt '):
                index_next.vt_index += 1
                text += line

        if g_mode:
            write_geo(text, geo_name)

    return geo2material


def get_transform(array: list) -> np.ndarray:
    return np.array(array, dtype=np.float32).reshape(4, 4).transpose()


def flatten_matrix4(m: np.ndarray) -> list:
    return m.reshape(16).tolist()


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def check_dir(input_path: Path, output_path: Path):
    if not Path.exists(input_path):
        raise FileNotFoundError(f'Path {input_path} not found')
    if not Path.exists(output_path):
        Path.mkdir(output_path)


def convert_camera(camera: dict) -> (str, dict):
    position = np.array(camera['eye'], dtype=np.float32)
    look_at = np.array(camera['look'], dtype=np.float32)
    front = normalize(look_at - position)
    up = normalize(np.array(camera['up'], dtype=np.float32))

    camera_dict = {
        'type': 'Camera',
        'impl': 'Pinhole',
        'prop': {
            'position': position.tolist(),
            'front': front.tolist(),
            'up': up.tolist(),
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
    name = camera['name']
    camera_names.add(name)
    print(f'Camera {name} converted')

    return name, camera_dict


def convert_light(light: dict, name: str) -> (str, dict):
    transform = flatten_matrix4(get_transform(light['translationMatrix']))
    exposure = light['exposure']
    color = np.array(light['color'], dtype=np.float32)[:3] * exposure

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
                                'v': color.tolist(),
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
                    'impl': 'Matrix',
                    'prop': {
                        'm': transform,
                    }
                },
            }
        }
    else:
        print(f'Light type {light["type"]} not supported')
        return None, None

    light_names.add(name)
    print(f'Light {name} converted')

    return name, light_dict


def convert_material(material: dict, name: str, geo_name: str) -> (str, dict):
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

        # # TODO: ignore textures and displacement now
        # if material.get('displacementMap', '') != '':
        #     prop['normal_map'] = {
        #         'type': 'Texture',
        #         'impl': 'Image',
        #         'prop': {
        #             'file': material['displacementMap'],
        #         }
        #     }

        material_dict = {
            'type': 'Surface',
            'impl': 'Disney',
            'prop': prop
        }
    name = f'{geo_name}_{name}'

    assignment = material['assignment']
    for shape in assignment:
        shape2material_geo = re_shape2material[geo_name]
        shape = shape.replace('*', '[0-9a-zA-Z_]+')
        shape = f'^{shape}$'
        mateial_map_geo = material_map[geo_name]

        name_existed = shape2material_geo.get(shape, name)
        if name_existed != name and mateial_map_geo[name_existed] != material_dict:
            err = f'Material of shape "{shape}" already specified, ' \
                  f'"{shape2material_geo[shape]}" -> "{name}"'
            print(mateial_map_geo[name_existed])
            print(material_dict)
            raise ValueError(err)

        shape2material_geo[shape] = name
        mateial_map_geo[name] = material_dict

    material_names.add(name)
    print(f'Material {name} converted')

    return name, material_dict


def convert_geometry(input_project_dir: Path, output_project_dir: Path, geo_name: str):
    input_json_dir = input_project_dir / 'json' / geo_name
    output_json_dir = output_project_dir / 'json' / geo_name
    check_dir(input_json_dir, output_json_dir)
    input_geo_dir = input_project_dir / 'obj' / geo_name
    output_geo_dir = output_project_dir / 'obj' / geo_name
    check_dir(input_geo_dir, output_geo_dir)

    with open(input_json_dir / f'{geo_name}.json', 'r') as f:
        geo = json.load(f)
    assert geo['name'] == geo_name
    assert Path(geo['geomObjFile']) == Path(f'obj/{geo_name}/{geo_name}.obj')
    assert Path(geo['matFile']) == Path(f'json/{geo_name}/materials.json')

    # materials
    material_file = input_project_dir / geo['matFile']
    with open(material_file, 'r') as f:
        materials = json.load(f)
    materials_dict = {}
    for name, material in materials.items():
        name, material_dict = convert_material(material, name, geo_name)
        material_dict = {name: material_dict}
        materials_dict.update(material_dict)
    material_file = output_project_dir / geo['matFile']
    with open(material_file, 'w') as f:
        json.dump(materials_dict, f, indent=2)
    material_file = material_file.relative_to(output_json_dir)

    json2obj_path = os.path.relpath(
        output_project_dir / geo['geomObjFile'],
        output_json_dir)
    json2obj_path = str(json2obj_path).replace('\\', '/')
    geo_obj_dict = {
        'type': 'Shape',
        'impl': 'Mesh',
        'prop': {
            'file': json2obj_path,
            'surface': 'TODO',
        }
    }
    geo2material = split_obj(input_project_dir / geo['geomObjFile'], output_geo_dir)
    exit(0)

    geo_dict = {
        'import': [
            material_file
        ],
        geo['name']: {
            'type': 'Shape',
            'impl': 'Group',
            'prop': {
                'shapes': [

                ],
                'transform': {
                    'type': 'Transform',
                    'impl': 'Matrix',
                    'prop': {
                        'm': flatten_matrix4(get_transform(geo['transformMatrix'])),
                    }
                }
            }
        }
    }


def copy_texture(src: Path, dst: Path):
    check_dir(src, dst)
    for file in src.iterdir():
        if file.is_file():
            shutil.copy(file, dst)


def create_main_scene_file(scene_dir: Path):
    integrator = {
        'type': 'Integrator',
        'impl': 'MegaPath',
        'prop': {
            'depth': 12,
            'rr_depth': 5,
            'rr_threshold': 0.95,
            'sampler': {
                'type': 'Sampler',
                'impl': 'PMJ02BN',
            }
        }
    }
    cameras = []
    for camera_name in camera_names:
        cameras.append(f'@{camera_name}')
    shapes = []
    for shape_name in shape_names:
        shapes.append(f'@{shape_name}')
    imports = []

    json_dir = scene_dir / 'json'
    for d in json_dir.iterdir():
        for f in d.iterdir():
            if f.suffix == '.json':
                import_path = str(f.relative_to(scene_dir)).replace('\\', '/')
                imports.append(import_path)

    scene = {
        'import': imports,
        'render': {
            'cameras': cameras,
            'integrator': integrator,
            'shapes': shapes,
            'spectrum': {
                'type': 'Spectrum',
                'impl': 'sRGB',
            }
        }
    }

    json.dump(scene, open(scene_dir / 'scene.json', 'w'), indent=2)


def disney2luisa(input_path: Path, output_path: Path):
    shutil.rmtree(output_path)
    check_dir(input_path, output_path)

    check_dir(input_path / 'obj', output_path / 'obj')
    path_in = input_path / 'json'
    path_out = output_path / 'json'
    check_dir(path_in, path_out)

    geo_names = [x.name for x in path_in.iterdir()
                 if x.is_dir() and x.name != 'cameras' and x.name != 'lights']

    # cameras
    path_in = input_path / 'json' / 'cameras'
    path_out = output_path / 'json' / 'cameras'
    check_dir(path_in, path_out)
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
    check_dir(path_in, path_out)
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
        material_map[geo_name] = {}
        re_shape2material[geo_name] = {}

    for geo_name in geo_names:
        path_in = input_path / 'json' / geo_name
        path_out = output_path / 'json' / geo_name
        check_dir(path_in, path_out)

        # geometries
        convert_geometry(input_path, output_path, geo_name)

    json.dump(re_shape2material, open(output_path / 're_shape2material.json', 'w'), indent=2)
    json.dump(material_map, open(output_path / 'material_map.json', 'w'), indent=2)

    # # textures
    # copy_texture(input_path / 'textures', output_path / 'textures')

    create_main_scene_file(output_path)


if __name__ == '__main__':
    if len(argv) == 3:
        disney2luisa(Path(argv[1]).absolute(), Path(argv[2]).absolute())
    else:
        print('Usage: disney2luisa.py <input.json> <output.luisa>')
