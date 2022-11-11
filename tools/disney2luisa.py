import shutil
from pathlib2 import Path
from sys import argv
import os
import json
import numpy as np
import re
import multiprocessing


class ConverterInfo:
    def __init__(self,
                 input_project_dir: Path, output_project_dir: Path,
                 input_json_dir: Path, output_json_dir: Path,
                 input_geo_dir: Path, output_geo_dir: Path):
        self.input_project_dir = input_project_dir
        self.output_project_dir = output_project_dir
        self.input_json_dir = input_json_dir
        self.output_json_dir = output_json_dir
        self.input_geo_dir = input_geo_dir
        self.output_geo_dir = output_geo_dir

        self.material_map = {}

        self.re_shape2material = {}

        self.material_names = set()
        self.environment_name = None
        self.camera_names = set()
        self.shape_names = set()

    def copy(self):
        result = ConverterInfo(self.input_project_dir, self.output_project_dir,
                               self.input_json_dir, self.output_json_dir,
                               self.input_geo_dir, self.output_geo_dir)
        result.material_map = self.material_map.copy()
        result.re_shape2material = self.re_shape2material.copy()
        result.material_names = self.material_names.copy()
        result.environment_name = self.environment_name
        result.camera_names = self.camera_names.copy()
        result.shape_names = self.shape_names.copy()
        return result

    def merge(self, other):
        self.material_map.update(other.material_map)
        self.re_shape2material.update(other.re_shape2material)
        self.material_names.update(other.material_names)
        self.camera_names.update(other.camera_names)
        self.shape_names.update(other.shape_names)
        if self.environment_name is None:
            self.environment_name = other.environment_name
        elif other.environment_name is not None and self.environment_name != other.environment_name:
            raise Exception('Multiple environment names found')


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

        def __add__(self, other):
            v = GeometryIndex()
            v.v_index = self.v_index + other.v_index
            v.vt_index = self.vt_index + other.vt_index
            v.vn_index = self.vn_index + other.vn_index
            return v

        def __sub__(self, other):
            v = GeometryIndex()
            v.v_index = self.v_index - other.v_index
            v.vt_index = self.vt_index - other.vt_index
            v.vn_index = self.vn_index - other.vn_index
            return v

        def __str__(self):
            return f"v:{self.v_index}, vt:{self.vt_index}, vn:{self.vn_index}"

        def copy(self):
            v = GeometryIndex()
            v.v_index = self.v_index
            v.vt_index = self.vt_index
            v.vn_index = self.vn_index
            return v

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

    def merge_geo(geo_name: str, index_num: GeometryIndex, geo_index_nums: dict):
        if geo_name in geo_index_nums:
            geo_index_nums[geo_name] += index_num
        else:
            geo_index_nums[geo_name] = index_num

    def write_geo_line(line: str, name: str, geo_lines: dict):
        lines = geo_lines.get(name, [])
        lines.append(line)
        geo_lines[name] = lines
        if len(lines) >= 1000000:
            write_geo_force(name, geo_lines)

    def write_geo_force(name: str, geo_lines: dict):
        lines = geo_lines[name]
        if len(lines) == 0:
            return
        new_obj_file_path = output_dir / (name + '.obj')
        with open(new_obj_file_path, 'a') as f:
            f.writelines(lines)
        geo_lines[name] = []

    def move_default_to_geo(name: str, geo_lines: dict):
        write_geo_force('default', geo_lines)
        input_path = output_dir / 'default.obj'
        output_path = output_dir / (name + '.obj')
        # print(f'Trying to move {input_path} to {output_path}')
        if input_path == output_path:
            return
        with open(input_path, 'r') as input_f:
            with open(output_path, 'a') as output_f:
                while True:
                    data = input_f.read(65536)
                    if not data:
                        break
                    output_f.write(data)
        os.remove(input_path)

    with open(path, 'r') as f:
        g_mode = False
        index = GeometryIndex()
        index_next = GeometryIndex()
        geo_index_nums = {}
        geo_lines = {}
        geo_lines.setdefault('default', [])
        geo2material = {}
        geo_name = None

        for line in f:
            if line.startswith('g '):
                geo_name_new = line[2:].strip()
                # print(f'geo_name transform: {geo_name} -> {geo_name_new}')
                # print(f'g_mode: {g_mode}')
                if g_mode:
                    g_mode = False
                    index_num = index_next - index
                    merge_geo(geo_name, index_num, geo_index_nums)
                    index = index_next.copy()
                elif geo_name is not None:
                    g_mode = True
                    move_default_to_geo(geo_name_new, geo_lines)
                geo_name = geo_name_new
            elif line.startswith('f '):
                if geo_name in geo_index_nums:
                    face = reindex(line, index - geo_index_nums[geo_name])
                else:
                    face = reindex(line, index)
                write_geo_line(f'f {face}\n', geo_name, geo_lines)
            elif line.startswith('usemtl '):
                material = line[7:].strip()
                geo2material[geo_name] = material
            elif line.startswith('v '):
                index_next.v_index += 1
                write_geo_line(line, geo_name, geo_lines)
            elif line.startswith('vn '):
                index_next.vn_index += 1
                write_geo_line(line, geo_name, geo_lines)
            elif line.startswith('vt '):
                index_next.vt_index += 1
                write_geo_line(line, geo_name, geo_lines)

        geo_lines.pop('default')
        for name in geo_lines:
            write_geo_force(name, geo_lines)

    return geo2material


def get_transform(array: list) -> np.ndarray:
    return np.array(array, dtype=np.float32).reshape(4, 4).transpose()


def identity_list(dim: int) -> list:
    return [1.0 if i == j else 0.0 for i in range(dim) for j in range(dim)]


def flatten_matrix4(m: np.ndarray) -> list:
    return m.reshape(16).tolist()


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def check_dir(input_path: Path, output_path: Path):
    if not Path.exists(input_path):
        raise FileNotFoundError(f'Path {input_path} not found')
    if not Path.exists(output_path):
        Path.mkdir(output_path)


def convert_camera(converter_info: ConverterInfo, camera: dict) -> (str, dict):
    position = np.array(camera['eye'], dtype=np.float32)
    look_at = np.array(camera['look'], dtype=np.float32)
    front = normalize(look_at - position)
    up = normalize(np.array(camera['up'], dtype=np.float32))
    name = camera['name']
    width_div_height = camera['ratio']
    width = 1000
    height = int(width / width_div_height)
    hfov = camera['fov']
    vfov = 2 * np.arctan(np.tan(hfov / 2 * np.pi / 180) / width_div_height) * 180 / np.pi
    vfov = float(vfov)

    camera_dict = {
        'type': 'Camera',
        'impl': 'Pinhole',
        'prop': {
            'transform': {
                'type': 'Transform',
                'impl': 'View',
                'prop': {
                    'position': position.tolist(),
                    'front': front.tolist(),
                    'up': up.tolist(),
                }
            },
            'fov': vfov,
            'spp': 64,
            'film': {
                'type': 'Film',
                'impl': 'Color',
                'prop': {
                    'resolution': [
                        width,
                        height
                    ],
                    'exposure': 0,
                }
            },
            'file': f'../../outputs/{name}.exr',
            'filter': {
                'type': 'Filter',
                'impl': 'Gaussian',
                'prop': {
                    'radius': 1
                }
            }
        }
    }
    converter_info.camera_names.add(name)
    print(f'Camera {name} converted')

    return name, camera_dict


def convert_light(converter_info: ConverterInfo, light: dict, name: str) -> (str, dict):
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
                        },
                        'two_sided': True,
                    }
                },
                'visible': False,
            }
        }
        converter_info.shape_names.add(name)
    elif light['type'] == 'dome':
        env_map_path = os.path.relpath(
            converter_info.output_project_dir / light['envmapCamera'].lstrip('island/'),
            converter_info.output_json_dir / 'lights')
        light_dict = {
            'type': 'Environment',
            'impl': 'Spherical',
            'prop': {
                'emission': {
                    'type': 'Texture',
                    'impl': 'Image',
                    'prop': {
                        'file': str(env_map_path).replace('\\', '/'),
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
        if converter_info.environment_name is not None:
            raise ValueError(f'Only one environment is supported, but {converter_info.environment_name} and {name} are found')
        converter_info.environment_name = name
    else:
        print(f'Light type {light["type"]} not supported')
        return None, None

    print(f'Light {name} converted')

    return name, light_dict


def convert_material(converter_info: ConverterInfo, material: dict, name: str, geo_name: str) -> (str, dict):
    if name == 'hidden':
        material_dict = {
            'type': 'Surface',
            'impl': 'Null',
        }
    else:
        thin = material['type'] == 'thin'

        def make_constant_texture(v: list) -> dict:
            value = v
            if type(value) != list:
                value = [value]
            if len(value) > 3:
                value = v[:3]
            return {
                'type': 'Texture',
                'impl': 'Constant',
                'prop': {
                    'v': value,
                }
            }

        prop = {
            'thin': thin,
            'color': make_constant_texture(material['baseColor']),
            'metallic': make_constant_texture(material['metallic']),
            'eta': make_constant_texture(material['ior']),
            'roughness': make_constant_texture(material['roughness']),
            'specular_tint': make_constant_texture(material['specularTint']),
            'anisotropic': make_constant_texture(material['anisotropic']),
            'sheen': make_constant_texture(material['sheen']),
            'sheen_tint': make_constant_texture(material['sheenTint']),
            'clearcoat': make_constant_texture(material['clearcoat']),
            'clearcoat_gloss': make_constant_texture(material['clearcoatGloss']),
            'specular_trans': make_constant_texture(material['specTrans']),
        }
        if thin:
            prop['flatness'] = make_constant_texture(material['flatness'])
            prop['diffuse_trans'] = make_constant_texture(material['diffTrans'])

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
        shape2material_geo = converter_info.re_shape2material[geo_name]
        shape = shape.replace('*', '[0-9a-zA-Z_]+')
        shape = f'^{shape}$'
        mateial_map_geo = converter_info.material_map[geo_name]

        name_existed = shape2material_geo.get(shape, name)
        if name_existed != name and mateial_map_geo[name_existed] != material_dict:
            err = f'Material of shape "{shape}" already specified, ' \
                  f'"{shape2material_geo[shape]}" -> "{name}"'
            print(mateial_map_geo[name_existed])
            print(material_dict)
            raise ValueError(err)

        shape2material_geo[shape] = name
        mateial_map_geo[name] = material_dict

    converter_info.material_names.add(name)
    print(f'Material {name} converted')

    return name, material_dict


def convert_geometry(converter_info: ConverterInfo, geo_name: str) -> (ConverterInfo, str, dict):
    print(f'Start to convert geometry {geo_name}')

    check_dir(converter_info.input_json_dir / geo_name, converter_info.output_json_dir / geo_name)
    check_dir(converter_info.input_geo_dir / geo_name, converter_info.output_geo_dir / geo_name)

    with open(converter_info.input_json_dir / geo_name / f'{geo_name}.json', 'r') as f:
        geo = json.load(f)
    assert geo['name'] == geo_name
    assert Path(geo['geomObjFile']) == Path(f'obj/{geo_name}/{geo_name}.obj')
    assert Path(geo['matFile']) == Path(f'json/{geo_name}/materials.json')

    # materials
    material_file = converter_info.input_project_dir / geo['matFile']
    with open(material_file, 'r') as f:
        materials = json.load(f)
    materials_dict = {}
    for name, material in materials.items():
        name, material_dict = convert_material(converter_info=converter_info, material=material, name=name, geo_name=geo_name)
        materials_dict[name] = material_dict
    material_file = converter_info.output_project_dir / geo['matFile']
    with open(material_file, 'w') as f:
        json.dump(materials_dict, f, indent=2)
    material_file = os.path.relpath(material_file, converter_info.output_json_dir / geo_name)

    geo_obj_dict = {
        'type': 'Shape',
        'impl': 'Group',
        'prop': {
            'shapes': [],
            'transform': {
                'type': 'Transform',
                'impl': 'Matrix',
                'prop': {
                    'm': [],
                }
            }
        }
    }

    # geometry of itself
    if 'geomObjFile' in geo:
        geo_file = converter_info.input_project_dir / geo['geomObjFile']
        geo2material = split_obj(geo_file, converter_info.output_geo_dir / geo_name)
        for shape, material in geo2material.items():
            geo_file = converter_info.output_geo_dir / geo_name / (shape + '.obj')
            json2obj_path = os.path.relpath(
                geo_file,
                converter_info.output_json_dir / geo_name)
            obj_part = {
                'type': 'Shape',
                'impl': 'Mesh',
                'prop': {
                    'file': str(json2obj_path).replace('\\', '/'),
                    'surface': f'@{geo_name}_{material}',
                }
            }
            geo_obj_dict['prop']['shapes'].append(obj_part)

    # primitives
    # TODO

    geo_dict = {
        'import': [
            str(material_file).replace('\\', '/'),
        ],
    }

    # instanced copy
    instanced_copies = geo.get('instancedCopies', {}).copy()
    instanced_copies[geo_name] = {
        "transformMatrix": geo.get('transformMatrix', identity_list(4)),
        "name": geo_name,
    }
    instanced_copy_names = list(instanced_copies.keys())
    instanced_copy_transforms = [instanced_copies[copy_name]['transformMatrix'] for copy_name in instanced_copy_names]
    for copy_name, instanced_copy_transform in zip(instanced_copy_names, instanced_copy_transforms):
        assert copy_name == instanced_copies[copy_name]['name']
        instanced_copy_transform = flatten_matrix4(get_transform(instanced_copy_transform))
        geo_obj_dict['prop']['transform']['prop']['m'] = instanced_copy_transform
        geo_dict[copy_name] = geo_obj_dict.copy()
        converter_info.shape_names.add(copy_name)

    print(f'Geometry {geo_name} converted')

    return converter_info, geo_name, geo_dict


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
            'depth': 8,
            'rr_depth': 2,
            'rr_threshold': 0.95,
            'sampler': {
                'type': 'Sampler',
                'impl': 'PMJ02BN',
            }
        }
    }
    cameras = []
    for camera_name in converter_info.camera_names:
        cameras.append(f'@{camera_name}')
    shapes = []
    for shape_name in converter_info.shape_names:
        shapes.append(f'@{shape_name}')
    imports = []

    geo_names = [x.name for x in (scene_dir / 'json').iterdir()
                 if x.is_dir() and x.name != 'cameras' and x.name != 'lights']
    for d in [scene_dir / 'json' / 'cameras', scene_dir / 'json' / 'lights']:
        for f in d.iterdir():
            if f.suffix == '.json':
                import_path = str(f.relative_to(scene_dir)).replace('\\', '/')
                imports.append(import_path)
    for geo_name in geo_names:
        imports.append(f'json/{geo_name}/{geo_name}.json')

    scene = {
        'import': imports,
        'render': {
            'cameras': cameras,
            'integrator': integrator,
            'shapes': shapes,
            'spectrum': {
                'type': 'Spectrum',
                'impl': 'sRGB',
            },
        }
    }
    if converter_info.environment_name is not None:
        scene['render']['environment'] = f'@{converter_info.environment_name}'

    json.dump(scene, open(scene_dir / 'scene.json', 'w'), indent=2)


def disney2luisa(input_project_dir: Path, output_project_dir: Path):
    shutil.rmtree(output_project_dir)
    check_dir(input_project_dir, output_project_dir)

    input_json_dir = input_project_dir / 'json'
    output_json_dir = output_project_dir / 'json'
    input_geo_dir = input_project_dir / 'obj'
    output_geo_dir = output_project_dir / 'obj'
    global converter_info
    converter_info = ConverterInfo(input_project_dir, output_project_dir, input_json_dir, output_json_dir, input_geo_dir, output_geo_dir)

    check_dir(converter_info.input_geo_dir, converter_info.output_geo_dir)
    path_in = converter_info.input_json_dir
    path_out = converter_info.output_json_dir
    check_dir(path_in, path_out)

    geo_names = [x.name for x in path_in.iterdir()
                 if x.is_dir() and x.name != 'cameras' and x.name != 'lights']

    # cameras
    path_in = converter_info.input_json_dir / 'cameras'
    path_out = converter_info.output_json_dir / 'cameras'
    check_dir(path_in, path_out)
    for camera_file in path_in.iterdir():
        with open(camera_file, 'r') as f:
            camera = json.load(f)
        name, camera_dict = convert_camera(converter_info=converter_info, camera=camera)
        camera_dict = {name: camera_dict}
        with open(path_out / camera_file.name, 'w') as f:
            json.dump(camera_dict, f, indent=2)

    # lights
    path_in = converter_info.input_json_dir / 'lights'
    path_out = converter_info.output_json_dir / 'lights'
    check_dir(path_in, path_out)
    quad = '''
v   1.00   1.00   0.00
v  -1.00   1.00   0.00
v  -1.00  -1.00   0.00
v   1.00  -1.00   0.00
f   -4   -3   -2   -1
'''
    with open(path_out / 'quad.obj', 'w') as f:
        f.write(quad)
    for light_file in path_in.iterdir():
        lights_dict = {}
        with open(light_file, 'r') as f:
            lights = json.load(f)
        for name, light in lights.items():
            name, light_dict = convert_light(converter_info=converter_info, light=light, name=name)
            if name is None:
                continue
            lights_dict[name] = light_dict
        with open(path_out / light_file.name, 'w') as f:
            json.dump(lights_dict, f, indent=2)

    # geometries
    for geo_name in geo_names:
        converter_info.material_map[geo_name] = {}
        converter_info.re_shape2material[geo_name] = {}

    test_geo_names = ['isBeach', 'isCoastline', 'osOcean', 'isDunesA', 'isDunesB', 'isMountainA', 'isMountainB']

    pool = multiprocessing.Pool()
    threads = []
    for geo_name in geo_names:
        # DEBUG
        # if geo_name not in test_geo_names:
        #     continue
        # geometries
        threads.append(pool.apply_async(convert_geometry, args=(converter_info.copy(), geo_name)))

    pool.close()
    pool.join()

    for thread_t in threads:
        converter_info_t, geo_name, geo_dict = thread_t.get()
        converter_info.merge(converter_info_t)
        path_in = converter_info.input_json_dir / geo_name
        path_out = converter_info.output_json_dir / geo_name
        check_dir(path_in, path_out)
        with open(path_out / f'{geo_name}.json', 'w') as f:
            json.dump(geo_dict, f, indent=2)

    json.dump(converter_info.re_shape2material, open(converter_info.output_project_dir / 're_shape2material.json', 'w'), indent=2)
    json.dump(converter_info.material_map, open(converter_info.output_project_dir / 'material_map.json', 'w'), indent=2)

    # textures
    copy_texture(input_project_dir / 'textures', converter_info.output_project_dir / 'textures')

    create_main_scene_file(converter_info.output_project_dir)


if __name__ == '__main__':
    if len(argv) == 3:
        disney2luisa(Path(argv[1]).absolute(), Path(argv[2]).absolute())
    else:
        print('Usage: disney2luisa.py <input.json> <output.luisa>')
