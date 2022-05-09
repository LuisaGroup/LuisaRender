import json
from pathlib import Path
from sys import argv
import glm
from xml.etree.ElementTree import *


def load_integer(context: Element) -> (str, int):
    if context.tag == 'integer':
        return context.attrib['name'], int(context.attrib['value'])


def load_string(context: Element) -> (str, str):
    if context.tag == 'string':
        return context.attrib['name'], context.attrib['value']


def load_bool(context: Element) -> (str, bool):
    if context.tag == 'boolean':
        return context.attrib['name'], context.attrib['value'].lower() == 'true'


def load_float(context: Element) -> (str, float):
    if context.tag == 'float':
        return context.attrib['name'], float(context.attrib['value'])


def load_value(context: Element):
    if context.tag == 'integer':
        return load_integer(context)
    elif context.tag == 'string':
        return load_string(context)
    elif context.tag == 'boolean':
        return load_bool(context)
    elif context.tag == 'float':
        return load_float(context)
    else:
        return None


def load_values(context: Element) -> dict:
    values_tag = {
        'integer',
        'string',
        'boolean',
        'float',
    }
    values = {}
    for child in context:
        if child.tag in values_tag:
            value = load_value(child)
            values[value[0]] = value[1]
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
        },
        'type': {
            'roughconductor': 'rough_conductor',
            'diffuse': 'lambert',
        },
    }
    bsdf_remove = [
        'ext_ior',
    ]

    keys = list(material.keys())
    for name_mitsuba2 in keys:
        if name_mitsuba2 in bsdf_remove:
            material.pop(name_mitsuba2)
        else:
            material[bsdf_map['key'].get(name_mitsuba2, name_mitsuba2)] = material.pop(name_mitsuba2)
    if 'type' in material:
        material['type'] = bsdf_map['type'].get(material['type'], material['type'])

    return material


def load_shape(context: Element) -> dict:
    return {}


def load_emitter(context: Element) -> dict:
    return {}


def load_camera(context: Element) -> dict:
    values = load_values(context)
    film = None
    film = load_values(context.find('./film'))
    if film is None:
        raise Exception('lacking film')
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
