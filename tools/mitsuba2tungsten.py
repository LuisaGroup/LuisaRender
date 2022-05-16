import json
import os
from pathlib import Path
from sys import argv
import glm
from xml.etree.ElementTree import *

scene_dict = {}
ref_dict = {}
bsdf_index = 0


def set_ref(context: Element, dict_node: dict):
    global ref_dict
    if 'id' in context.attrib:
        ref_dict[context.attrib['id']] = dict_node


def get_ref(name: str) -> str:
    global ref_dict
    name = name.strip(' ')
    if name.startswith('$'):
        name = name[1:]
    return ref_dict.get(name, name)


def load_integer(context: Element) -> (str, int):
    return context.attrib['name'], int(get_ref(context.attrib['value']))


def load_string(context: Element) -> (str, str):
    return context.attrib['name'], get_ref(context.attrib['value'])


def load_bool(context: Element) -> (str, bool):
    return context.attrib['name'], get_ref(context.attrib['value']).lower() == 'true'


def load_float(context: Element) -> (str, float):
    return context.attrib['name'], float(get_ref(context.attrib['value']))


def load_rgb(context: Element) -> (str, list):
    numbers = context.attrib['value'].split(',')
    assert len(numbers) == 3
    values = load_values(context)
    scale = values.get('scale', 1.0)
    for i in range(len(numbers)):
        numbers[i] = float(get_ref(numbers[i])) * scale
    return context.attrib['name'], numbers


def load_spectrum(context: Element) -> (str, list):
    numbers = context.attrib['value'].split(',')
    assert len(numbers) == 1
    values = load_values(context)
    scale = values.get('scale', 1.0)
    for i in range(len(numbers)):
        numbers[i] = float(get_ref(numbers[i])) * scale
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
            matrix[i][j] = float(get_ref(value[i * 4 + j]))
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
    elif context.tag == 'spectrum':
        return load_spectrum(context)
    elif context.tag == 'matrix':
        return load_matrix(context)
    else:
        return None


def load_look_at(context: Element) -> dict:
    assert len(list(context.attrib)) == 3
    data = [
        context.attrib['origin'].split(','),
        context.attrib['target'].split(','),
        context.attrib['up'].split(','),
    ]
    global ref_dict
    print(ref_dict)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(get_ref(data[i][j]))

    return {
        'position': data[0],
        'look_at': data[1],
        'up': data[2],
    }


def load_translate(context: Element) -> dict:
    assert len(list(context.attrib)) == 3
    return {
        'position': [
            float(get_ref(context.attrib['x'])),
            float(get_ref(context.attrib['y'])),
            float(get_ref(context.attrib['z'])),
        ],
        'scale': 1.,
        'rotation': [
            0.,
            0.,
            0.,
        ],
    }


def load_transform(context: Element) -> dict:
    transform = {}
    assert len(list(context)) == 1

    if context[0].tag == 'matrix':
        # TODO
        raise Exception('matrix unsupported')

        matrix = load_value(context[0])[1]
        if matrix != [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]:
            transform = {'matrix': matrix}
    elif context[0].tag == 'lookat':
        transform = load_look_at(context[0])
    elif context[0].tag == 'translate':
        transform = load_translate(context[0])
    else:
        raise Exception(f'Unknown transform "{context[0].tag}"')

    return transform


def load_values(context: Element) -> dict:
    global ref_dict
    values_tag = {'integer', 'string', 'boolean', 'float', 'rgb', 'matrix', 'spectrum'}
    skip_tag = {'sampler', 'rfilter', 'bsdf'}

    values = {}
    for child in context:
        if child.tag in values_tag:
            value = load_value(child)
            values[value[0]] = value[1]
        elif child.tag == 'ref':
            values['ref'] = child.attrib['id']
        elif child.tag == 'transform':
            values['transfrom'] = load_transform(child)
        elif child.tag == 'emitter':
            values.update(load_emitter(child))
        elif child.tag == 'film':
            values['film'] = load_values(child)
            if 'id' in child.attrib:
                ref_dict[child.attrib['id']] = values['film']
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
        # ref bsdf
        shape['bsdf'] = shape.pop('ref')
    elif shape.get('bsdf', None) is None:
        material_node = context.find('bsdf')
        if material_node is None:
            raise Exception('primitive lacking bsdf')
        else:
            # bsdf inline
            material = load_material(material_node)

            # no name, create one
            if material.get('name', None) is None:
                # no name
                global bsdf_name, bsdf_index
                name = f'mat_{bsdf_index}'
                while name in bsdf_name:
                    bsdf_index += 1
                    name = f'mat_{bsdf_index}'
                material['name'] = name
                bsdf_name.add(name)

            global scene_dict
            scene_dict['bsdfs'].append(material)
            shape['bsdf'] = material['name']

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
    global ref_dict
    values = load_values(context)

    film = values.get('film', None)
    for iter in context.findall('ref'):
        name = iter.attrib['id']
        ref_dict_node = ref_dict.get(name, None)
        if (not ref_dict_node is None) and 'width' in ref_dict_node:
            film = ref_dict_node
            break
    if film is None:
        raise Exception('No film in camera')

    camera = {
        "tonemap": "filmic",
        "resolution": [
            film['width'],
            film['height']
        ],
        "reconstruction_filter": "tent",
        "transform": load_transform(context.find('transform')),
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
    global scene_dict, ref_dict, bsdf_index
    ref_dict = {}
    bsdf_index = 0
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
            name = child.attrib['name']
            ref_dict[name] = child.attrib['value']
        elif child.tag == 'integrator':
            scene_dict['integrator'] = load_integrator(child)
        elif child.tag == 'sensor':
            scene_dict['camera'] = load_camera(child)
        elif child.tag == 'bsdf':
            material = load_material(child)
            ref_dict[material['name']] = material
            scene_dict['bsdfs'].append(material)
        elif child.tag == 'shape':
            scene_dict['primitives'].append(load_shape(child))
        elif child.tag == 'emitter':
            scene_dict['primitives'].append(load_emitter(child))
        elif child.tag == 'integer':
            pass
        elif child.tag == 'film':
            ref_dict[child.attrib['id']] = load_values(child)
        elif child.tag == 'sampler':
            pass
        else:
            raise Exception(f'Unexpected node "{child.tag}"')
    return scene_dict


def expand_node(context: Element):
    index = 0
    while index < len(list(context)):
        child = context[index]
        if child.tag == 'include':
            sub_file_name = child.attrib['filename']
            context.remove(child)
            _, sub_node = load_element(sub_file_name)
            for sub_node_child in sub_node:
                context.insert(index, sub_node_child)
                index += 1
        else:
            print(child.tag)
            expand_node(context[index])
            index += 1


def load_element(file_name: str) -> (ElementTree, Element):
    assert file_name.endswith(".xml")
    tree = ElementTree(file=file_name)
    context = tree.getroot()
    expand_node(context)
    return tree, context


def mitsuba2json(file_name: str):
    os.chdir(os.path.dirname(os.path.realpath(file_name)))
    scene_tree, scene_root = load_element(file_name)
    scene_tree.write(file_name[:-4] + '_expanded.xml')

    with open(f"{file_name[:-4]}.json", "w") as file:
        json.dump(load_root(scene_root), file, indent=4)


if __name__ == '__main__':
    file_name = argv[1].strip('"').strip(' ')
    mitsuba2json(file_name)
