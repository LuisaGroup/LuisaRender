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


def load_material(context: Element) -> dict:
    return {}


def load_shape(context: Element) -> dict:
    return {}


def load_emitter(context: Element) -> dict:
    return {}


def load_camera(context: Element) -> dict:
    return {}


def load_integrator(context: Element) -> dict:
    max_depth = 16
    for child in context:
        if child.tag == 'integer' and child.attrib['name'] == 'max_depth':
            max_depth = int(child.attrib['value'])
            break
    integrator = {
        "min_bounces": 0,
        "max_bounces": max_depth,
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
