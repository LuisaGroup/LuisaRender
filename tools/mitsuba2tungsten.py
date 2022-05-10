import json
from pathlib import Path
from sys import argv
import glm
from xml.etree.ElementTree import *


def load_integer(context: Element) -> (str, int):
    return context.attrib['name'], int(context.attrib['value'])


def load_string(context: Element) -> (str, str):
    return context.attrib['name'], context.attrib['value']


def load_bool(context: Element) -> (str, bool):
    return context.attrib['name'], context.attrib['value'].lower() == 'true'


def load_float(context: Element) -> (str, float):
    return context.attrib['name'], float(context.attrib['value'])


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


def load_sub_material(context: Element) -> dict:
    material = load_values(context)

    if context.attrib['type'] != 'twosided':
        material['type'] = context.attrib['type']

    return material


def load_material(context: Element) -> dict:
    material = {'name': context.attrib['id']}

    if context.attrib['type'] != 'twosided':
        material['type'] = context.attrib['type']

    material.update(load_sub_material(context))

    if context[0].tag == 'bsdf':
        assert len(list(context)) <= 1
        material.update(load_sub_material(context[0]))

    # wash data
    bsdf_map = {
        'key': {
            'alpha': 'roughness',
            'int_ior': 'ior',
        },
        'type': {
            'roughconductor': 'rough_conductor',
            'roughplastic': 'rough_plastic',
            'diffuse': 'lambert',
        },
    }
    bsdf_remove = ['ext_ior', 'nonlinear']
    keys = list(material.keys())
    for name_mitsuba2 in keys:
        if name_mitsuba2 in bsdf_remove:
            material.pop(name_mitsuba2)
        elif name_mitsuba2 in bsdf_map['key']:
            material[bsdf_map['key'][name_mitsuba2]] = material.pop(name_mitsuba2)
    if 'type' in material and material['type'] in bsdf_map['type']:
        material['type'] = bsdf_map['type'][material['type']]

    return material


def load_shape(context: Element) -> dict:
    shape_map = {
        'key': {
            'filename': 'file',
            'ref': 'bsdf',
        },
        'type': {
            'obj': 'mesh',
            'rectangle': 'quad',
        },
    }
    shape_remove = ['face_normals']

    shape_type = shape_map['type'][context.attrib['type']]
    shape = {
        "transform": {},
        "type": shape_type,
        "smooth": False,
        "backface_culling": False,
        "recompute_normals": False,
        "file": "",
        "bsdf": "",
    }
    shape.update(load_values(context))

    # wash data
    keys = list(shape.keys())
    for name_mitsuba2 in keys:
        if name_mitsuba2 in shape_remove:
            shape.pop(name_mitsuba2)
        elif name_mitsuba2 in shape_map['key']:
            shape[shape_map['key'][name_mitsuba2]] = shape.pop(name_mitsuba2)

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


def load_emitter(context: Element) -> dict:
    emitter_type = context.attrib['type']

    if emitter_type == 'constant':
        return load_constant_emitter(context)
    elif emitter_type == 'area':
        # TODO
        return {}
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
        print(child.tag, child.attrib)
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
