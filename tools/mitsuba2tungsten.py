import json
from pathlib import Path
from sys import argv
import glm
from xml.etree.ElementTree import *

variables = {}


def get_variable(name: str) -> str:
    global variables
    if name.startswith('$') and name[1:] in variables:
        return variables[name[1:]]
    else:
        return name


def load_integer(context: Element) -> (str, int):
    return context.attrib['name'], int(get_variable(context.attrib['value']))


def load_string(context: Element) -> (str, str):
    return context.attrib['name'], get_variable(context.attrib['value'])


def load_bool(context: Element) -> (str, bool):
    return context.attrib['name'], get_variable(context.attrib['value']).lower() == 'true'


def load_float(context: Element) -> (str, float):
    return context.attrib['name'], float(get_variable(context.attrib['value']))


def load_rgb(context: Element) -> (str, list):
    numbers = context.attrib['value'].split(',')
    for i in range(len(numbers)):
        numbers[i] = float(numbers[i])
    return context.attrib['name'], numbers


def load_matrix(context: Element) -> (str, list):
    value = context.attrib['value'].split(' ')
    assert len(value) == 16
    matrix = [
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
    ]
    for i in range(4):
        for j in range(4):
            matrix[i][j] = float(value[i * 4 + j])
    return 'matrix', matrix


def load_value(context: Element):
    if context.tag == 'integer':
        return load_integer(context)
    elif context.tag == 'string':
        return load_string(context)
    elif context.tag == 'boolean':
        return load_bool(context)
    elif context.tag == 'float':
        return load_float(context)
    elif context.tag == 'rgb':
        return load_rgb(context)
    elif context.tag == 'matrix':
        return load_matrix(context)
    else:
        return None


def load_values(context: Element) -> dict:
    values_tag = {'integer', 'string', 'boolean', 'float', 'rgb', 'matrix'}
    skip_tag = {'sampler', 'rfilter', 'bsdf'}

    values = {}
    for child in context:
        if child.tag in values_tag:
            value = load_value(child)
            values[value[0]] = value[1]
        elif child.tag == 'ref':
            values['ref'] = child.attrib['id']
        elif child.tag == 'transform':
            assert len(list(child)) == 1 and child[0].tag == 'matrix'
            matrix = load_value(child[0])[1]
            if matrix == [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ]:
                values['transform'] = {}
            else:
                values['transform'] = {'matrix': matrix}
        elif child.tag == 'emitter':
            values.update(load_emitter(child))
        elif child.tag == 'film':
            values['film'] = load_values(child)
        elif child.tag in skip_tag:
            pass
        else:
            raise Exception(f'parse exception "{child.tag}"')

    return values


def load_diffuse(material: dict):
    material['type'] = 'lambert'
    material['albedo'] = material.pop('reflectance')


def load_roughconductor(material: dict):
    material['type'] = 'rough_conductor'
    material['albedo'] = material.pop('specular_reflectance')
    material['roughness'] = material.pop('alpha')


def load_roughdielectric(material: dict):
    material['type'] = 'rough_dielectric'
    material['albedo'] = 1.0
    material['roughness'] = material.pop('alpha')
    material['ior'] = material.pop('int_ior')
    material['enable_refraction'] = True


def load_roughplastic(material: dict):
    material['type'] = 'rough_plastic'
    material['albedo'] = material.pop('diffuse_reflectance')
    material['roughness'] = material.pop('alpha')
    material['ior'] = material.pop('int_ior')
    material['thickness'] = 1.0
    material['sigma_a'] = 0.0

    key_remove = ['nonlinear', 'ext_ior']
    for key in key_remove:
        material.pop(key, None)


def load_material(context: Element) -> dict:
    material = {}
    if 'id' in context.attrib:
        material['name'] = context.attrib['id']

    if context.attrib['type'] != 'twosided':
        material['type'] = context.attrib['type']

    material.update(load_values(context))

    if len(list(context)) > 0:
        child = context.find('bsdf')
        if not child is None:
            material.update(load_values(child))
            if child.attrib['type'] != 'twosided':
                material['type'] = child.attrib['type']

    if material['type'] == 'diffuse':
        load_diffuse(material)
    elif material['type'] == 'roughconductor':
        load_roughconductor(material)
    elif material['type'] == 'roughdielectric':
        load_roughdielectric(material)
    elif material['type'] == 'roughplastic':
        load_roughplastic(material)
    else:
        raise Exception(f'Unknown material "{material["type"]}"')

    return material


def load_obj_shape(shape: dict):
    shape['type'] = 'mesh'
    shape['smooth'] = False
    shape['backface_culling'] = False
    shape['recompute_normals'] = False
    shape['file'] = shape.pop('filename')
    shape.pop('face_normals', None)


def load_rectangle_shape(shape: dict):
    shape['type'] = 'quad'


def load_shape(context: Element) -> dict:
    shape = {}
    shape.update(load_values(context))

    if context.attrib['type'] == 'obj':
        load_obj_shape(shape)
    elif context.attrib['type'] == 'rectangle':
        load_rectangle_shape(shape)
    else:
        raise Exception(f'Unknown shape type "{context.attrib["type"]}"')

    if 'ref' in shape:
        shape['bsdf'] = shape.pop('ref')
    elif shape.get('bsdf', None) is None:
        bsdf = context.find('bsdf')
        if bsdf is None:
            raise Exception('primitive lacking bsdf')
        else:
            shape['bsdf'] = load_material(bsdf)

    return shape


def load_constant_emitter(context: Element) -> dict:
    values = load_values(context)
    emitter = {
        "transform": {},
        "emission": values['radiance'],
        "type": "infinite_sphere",
        "sample": False
    }
    return emitter


def load_area_emitter(context: Element) -> dict:
    emitter = load_values(context)
    emitter['emission'] = emitter.pop('radiance')
    return emitter


def load_emitter(context: Element) -> dict:
    emitter_type = context.attrib['type']

    if emitter_type == 'constant':
        return load_constant_emitter(context)
    elif emitter_type == 'area':
        return load_area_emitter(context)
    else:
        raise Exception(f'Unknown emitter type "{emitter_type}"')


def load_camera(context: Element) -> dict:
    values = load_values(context)
    film = values['film']
    camera = {
        "tonemap": "filmic",
        "resolution": [
            film['width'],
            film['height']
        ],
        "reconstruction_filter": "tent",
        "transform": {
            "position": [
                -0.5196635723114014,
                0.8170070052146912,
                3.824389696121216
            ],
            "look_at": [
                -0.0668703019618988,
                0.6448959708213806,
                0.5292789936065674
            ],
            "up": [
                0.0,
                1.0,
                0.0
            ]
        },
        "type": "pinhole",
        "fov": values['fov']
    }
    # TODO: camera settings
    return camera


def load_integrator(context: Element) -> dict:
    values = load_values(context)
    integrator = {
        "min_bounces": 0,
        "max_bounces": values['max_depth'],
        "enable_consistency_checks": False,
        "enable_two_sided_shading": True,
        "type": "path_tracer",
        "enable_light_sampling": True,
        "enable_volume_light_sampling": True
    }
    return integrator


def load_root(context: Element) -> dict:
    scene_dict = {
        'media': [],
        'bsdfs': [],
        'primitives': [],
        'camera': {},
        'integrator': {},
        "renderer": {
            "overwrite_output_files": True,
            "adaptive_sampling": True,
            "enable_resume_render": False,
            "stratified_sampler": True,
            "scene_bvh": True,
            "spp": 64,
            "spp_step": 16,
            "checkpoint_interval": "0",
            "timeout": "0",
            "output_file": "output.png",
            "resume_render_file": "TungstenRenderState.dat",
            "hdr_output_file": "output.exr"
        }
    }
    for child in context:
        if child.tag == 'default':
            variables[child.attrib['name']] = child.attrib['value']
        if child.tag == 'integrator':
            scene_dict['integrator'] = load_integrator(child)
        elif child.tag == 'sensor':
            scene_dict['camera'] = load_camera(child)
        elif child.tag == 'bsdf':
            scene_dict['bsdfs'].append(load_material(child))
        elif child.tag == 'shape':
            scene_dict['primitives'].append(load_shape(child))
        elif child.tag == 'emitter':
            scene_dict['primitives'].append(load_emitter(child))
        else:
            raise Exception(f'Unexpected node "{child.tag}"')
    return scene_dict


def mitsuba2json(file_name: str):
    assert file_name.endswith(".xml")
    scene_tree = ElementTree(file=file_name)
    scene_root = scene_tree.getroot()

    with open(f"{file_name[:-4]}.json", "w") as file:
        json.dump(load_root(scene_root), file, indent=4)


if __name__ == '__main__':
    file_name = argv[1].strip('"').strip(' ')
    mitsuba2json(file_name)
