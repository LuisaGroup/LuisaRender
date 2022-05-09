import json
from pathlib import Path
from sys import argv
import glm
from xml.etree.ElementTree import *


def load_materials(scene_text) -> dict:
    pass


def load_shapes(scene_text) -> dict:
    pass


def load_camera(scene_text) -> dict:
    pass


def mitsuba2json(file_name: str, spp: int):
    assert file_name.endswith(".xml")
    assert (spp % 16) == 0
    scene = ElementTree(file=file_name)
    print(scene)
    print(scene.getroot())
    exit(0)

    materials = load_materials(scene)
    shapes = load_shapes(scene)
    camera = load_camera(scene)
    print(materials)
    print(shapes)
    print(camera)
    scene_dict = {
        'media': [],
        'bsdfs': materials,
        'primitives': shapes,
        'camera': camera,
        'integrator': {
            "min_bounces": 0,
            "max_bounces": 16,
            "enable_consistency_checks": False,
            "enable_two_sided_shading": True,
            "type": "path_tracer",
            "enable_light_sampling": True,
            "enable_volume_light_sampling": True
        },
        "renderer": {
            "overwrite_output_files": True,
            "adaptive_sampling": True,
            "enable_resume_render": False,
            "stratified_sampler": True,
            "scene_bvh": True,
            "spp": spp,
            "spp_step": 16,
            "checkpoint_interval": "0",
            "timeout": "0",
            "output_file": "output.png",
            "resume_render_file": "TungstenRenderState.dat",
            "hdr_output_file": "output.exr"
        }
    }
    with open(f"{file_name[:-4]}.json", "w") as file:
        json.dump(scene_dict, file, indent=4)


if __name__ == '__main__':
    file_name = argv[1].strip('"').strip(' ')
    spp = int(argv[2])
    mitsuba2json(file_name, spp)
