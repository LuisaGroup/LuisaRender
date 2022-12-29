# convert [GLSL-PathTracer](https://github.com/knightcrawler25/GLSL-PathTracer) scenes to LuisaRender


import glm
import json
import sys
import math


def do_conversion(scene_name, nodes):
    textures = {}
    surfaces = {}
    meshes = {}
    emissive_surfaces = {}
    render = {
        "integrator": {
            "impl": "WavePath",
            "prop": {
                "sampler": {
                    "impl": "PMJ02BN"
                }
            }
        },
        "cameras": [],
        "shapes": []
    }
    spp = 1024
    resolution = [1920, 1080]
    env_scale = 1.0
    hide_emissive = False
    for tag, prop in nodes:
        if tag == "renderer":
            for k, v in prop.items():
                if k == "envmapfile":
                    if v[0] == "none":
                        continue
                    render["environment"] = {
                        "impl": "Spherical",
                        "prop": {
                            "emission": {
                                "impl": "Image",
                                "prop": {
                                    "file": v[0]
                                }
                            },
                            "transform": {
                                "impl": "SRT",
                                "prop": {
                                    "rotate": [0, 1, 0, -90]
                                }
                            }
                        }
                    }
                elif k == "resolution":
                    resolution = [int(v[0]), int(v[1])]
                elif k == "envmapintensity":
                    env_scale = float(v[0])
                elif k == "hideemitters":
                    hide_emissive = (v[0] == "true")
        elif tag == "mesh":
            file = None
            material = None
            transform_matrix = None
            translation = None
            rotation = None
            scale = None
            for k, v in prop.items():
                if k == "file":
                    file = v[0]
                elif k == "material":
                    material = v[0]
                elif k == "matrix":
                    transform_matrix = [float(x) for x in v]
                elif k == "position":
                    translation = [float(x) for x in v]
                elif k == "rotation":
                    quat = glm.quat([float(x) for x in v])
                    angle = glm.degrees(2 * glm.acos(quat.w))
                    axis = glm.normalize(glm.vec3(quat.x, quat.y, quat.z))
                    rotation = [axis.x, axis.y, axis.z, angle]
                elif k == "scale":
                    scale = [float(x) for x in v]
            assert file is not None and material is not None
            if file not in meshes:
                meshes[file] = {
                    "type": "Shape",
                    "impl": "Mesh",
                    "prop": {
                        "file": file,
                        "shadow_terminator": 1.0
                    }
                }
            shape = {
                "impl": "Instance",
                "prop": {
                    "shape": f"@Mesh:{file}",
                    "surface": f"@Surface:{material}",
                    "shadow_terminator": 1.0
                }
            }
            if material in emissive_surfaces:
                shape["prop"]["light"] = f"@Light:{material}"
            if transform_matrix:
                shape["prop"]["transform"] = {
                    "impl": "Matrix",
                    "prop": {
                        "m": transform_matrix
                    }
                }
            elif translation or rotation or scale:
                shape["prop"]["transform"] = {
                    "impl": "SRT",
                    "prop": {
                        "translate": translation or [0, 0, 0],
                        "rotate": rotation or [0, 1, 0, 0],
                        "scale": scale or [1, 1, 1]
                    }
                }
            render["shapes"].append(shape)
        elif tag == "material":
            surface = {
                "type": "Surface",
                "impl": "Disney",
                "prop": {}
            }
            name = prop["name"][0]

            def get_texture(name, linear=True):
                if name.endswith(".exr") or name.endswith(".hdr"):
                    linear = True
                if name not in textures:
                    textures[name] = {
                        "type": "Texture",
                        "impl": "Image",
                        "prop": {
                            "file": name,
                            "encoding": "sRGB" if not linear else "Linear"
                        }
                    }
                return f"@Texture:{name}"

            for k, v in prop.items():
                if k == "color":
                    if "color" in surface["prop"]:
                        continue
                    surface["prop"]["color"] = {
                        "impl": "Constant",
                        "prop": {"v": [float(x) for x in v]}
                    }
                elif k == "opacity":
                    if "alpha" in surface["prop"]:
                        continue
                    surface["prop"]["alpha"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "metallic":
                    if "metallic" in surface["prop"]:
                        continue
                    surface["prop"]["metallic"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "roughness":
                    if "roughness" in surface["prop"]:
                        continue
                    surface["prop"]["roughness"] = {
                        "impl": "Constant",
                        "prop": {"v": glm.sqrt(float(v[0]))}
                    }
                elif k == "subsurface":
                    if "subsurface" in surface["prop"]:
                        continue
                    surface["prop"]["subsurface"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "speculartint":
                    if "specular_tint" in surface["prop"]:
                        continue
                    surface["prop"]["specular_tint"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "anisotropic":
                    if "anisotropic" in surface["prop"]:
                        continue
                    surface["prop"]["anisotropic"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "sheen":
                    if "sheen" in surface["prop"]:
                        continue
                    surface["prop"]["sheen"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "sheentint":
                    if "sheen_tint" in surface["prop"]:
                        continue
                    surface["prop"]["sheen_tint"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "clearcoat":
                    if "clearcoat" in surface["prop"]:
                        continue
                    surface["prop"]["clearcoat"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "clearcoatgloss":
                    if "clearcoat_gloss" in surface["prop"]:
                        continue
                    surface["prop"]["clearcoat_gloss"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "spectrans":
                    if "specular_trans" in surface["prop"]:
                        continue
                    surface["prop"]["specular_trans"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "ior":
                    if "eta" in surface["prop"]:
                        continue
                    surface["prop"]["eta"] = {
                        "impl": "Constant",
                        "prop": {"v": float(v[0])}
                    }
                elif k == "emission":
                    if name in emissive_surfaces:
                        continue
                    emissive_surfaces[name] = {
                        "type": "Light",
                        "impl": "Diffuse",
                        "prop": {
                            "emission": {
                                "impl": "Constant",
                                "prop": {"v": [float(x) for x in v]}
                            }
                        }
                    }
                elif k == "albedotexture":
                    surface["prop"]["color"] = get_texture(v[0], False)
                elif k == "metallicroughnesstexture":
                    t = get_texture(v[0])
                    surface["prop"]["metallic"] = {
                        "impl": "Swizzle",
                        "prop": {
                            "base": t,
                            "swizzle": 2
                        }
                    }
                    surface["prop"]["roughness"] = {
                        "impl": "Swizzle",
                        "prop": {
                            "base": t,
                            "swizzle": 1
                        }
                    }
                elif k == "normaltexture":
                    surface["prop"]["normal_map"] = get_texture(v[0])
                elif k == "emissiontexture":
                    emissive_surfaces[name] = {
                        "type": "Light",
                        "impl": "Diffuse",
                        "prop": {
                            "emission": get_texture(v[0], False)
                        }
                    }
                elif k != "name":
                    print(f"Unrecognized material property: {k}")
            if "roughness" not in surface["prop"]:
                surface["prop"]["roughness"] = {
                    "impl": "Constant",
                    "prop": {
                        "v": 1
                    }
                }
            surfaces[name] = surface
        elif tag == "light":
            assert prop["type"][0] == "quad" or prop["type"][0] == "sphere"
            if prop["type"][0] == "quad":
                p0 = glm.vec3([float(x) for x in prop["position"]])
                p1 = glm.vec3([float(x) for x in prop["v1"]])
                p3 = glm.vec3([float(x) for x in prop["v2"]])
                p2 = p3 + p1 - p0
                p = [p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z]
                t = [0, 1, 2, 0, 2, 3]
                render["shapes"].append({
                    "impl": "InlineMesh",
                    "prop": {
                        "indices": t,
                        "positions": p,
                        "light": {
                            "impl": "Diffuse",
                            "prop": {
                                "emission": {
                                    "impl": "Constant",
                                    "prop": {"v": [float(x) for x in prop["emission"]]}
                                }
                            }
                        }
                    }
                })
            else:  # sphere
                radius = float(prop["radius"][0])
                position = [float(x) for x in prop["position"]]
                emission = [float(x) for x in prop["emission"]]
                meshes["luisa-render-sphere"] = {
                    "type": "Shape",
                    "impl": "Sphere",
                    "prop": {
                        "subdivision": 4
                    }
                }
                render["shapes"].append({
                    "impl": "Instance",
                    "prop": {
                        "shape": "@Mesh:luisa-render-sphere",
                        "transform": {
                            "impl": "SRT",
                            "prop": {
                                "scale": radius,
                                "translate": position
                            }
                        },
                        "light": {
                            "impl": "Diffuse",
                            "prop": {
                                "emission": {
                                    "impl": "Constant",
                                    "prop": {"v": emission}
                                }
                            }
                        }
                    }
                })
        elif tag == "camera":
            position = None
            target = None
            fov = None
            matrix = None
            lens_diameter = 0.001
            focus_distance = None
            for k, v in prop.items():
                if k == "position":
                    position = [float(x) for x in v]
                elif k == "lookat":
                    target = [float(x) for x in v]
                elif k == "fov":
                    fov = float(v[0])
                elif k == "matrix":
                    # transpose the matrix
                    matrix = [[float(v[j * 4 + i]) for j in range(4)] for i in range(4)]
                elif k == "aperture":
                    lens_diameter = float(v[0])
                elif k == "focaldist":
                    focus_distance = float(v[0])
            assert matrix is not None or (position is not None and target is not None)
            assert fov is not None
            focal_length = 18 / glm.tan(glm.radians(fov) / 2)
            # hfov to vfov
            fov = glm.degrees(2 * glm.atan(glm.tan(glm.radians(fov) / 2) * resolution[1] / resolution[0]))
            aperture = focal_length / 100000 / lens_diameter
            if matrix is not None:
                position = [x for x in matrix[3][:3]]
                front = [x for x in matrix[2][:3]]
            else:
                front = [x for x in glm.normalize(glm.vec3(target) - glm.vec3(position))]
            if focus_distance:
                camera = {
                    "impl": "ThinLens",
                    "prop": {
                        "focal_length": focal_length,
                        "focus_distance": focus_distance,
                        "aperture": aperture
                    }
                }
            else:
                camera = {
                    "impl": "Pinhole",
                    "prop": {
                        "fov": fov
                    }
                }
            camera["prop"]["transform"] = {
                "impl": "View",
                "prop": {
                    "position": position,
                    "front": front
                }
            }
            render["cameras"].append(camera)
        else:
            raise NotImplementedError(f"Unsupported node type: {tag}")
    if hide_emissive:
        for s in render["shapes"]:
            if "light" in s["prop"]:
                s["prop"]["visible"] = False
    if "environment" in render:
        render["environment"]["prop"]["scale"] = env_scale
    if (r := max(*resolution)) < 1920:
        resolution = [int(round(x * 1920 / r)) for x in resolution]
    for i, camera in enumerate(cc := render["cameras"]):
        camera["prop"]["film"] = {
            "impl": "Color",
            "prop": {
                "resolution": resolution,
                "clamp": 64
            }
        }
        camera["prop"]["file"] = f"{scene_name}-view-{i}.exr" if len(cc) > 1 else f"{scene_name}.exr"
        camera["prop"]["filter"] = {
            "impl": "Gaussian",
            "prop": {
                "radius": glm.clamp(0.5 * round(min(*resolution) / 1024 * 3), 1, 3)
            }
        }
        camera["prop"]["spp"] = spp
    scene = {"render": render}
    for name, mesh in meshes.items():
        scene[f"Mesh:{name}"] = mesh
    for name, surface in surfaces.items():
        scene[f"Surface:{name}"] = surface
    for name, texture in textures.items():
        scene[f"Texture:{name}"] = texture
    for name, light in emissive_surfaces.items():
        scene[f"Light:{name}"] = light
    return scene


def convert_scene(path):
    assert path.endswith(".scene")
    with open(path, 'r') as file:
        lines = [l.strip().split() for l in file.readlines()]
    nodes = []
    curr_node = None
    for line in lines:
        if not line or line[0].startswith("#"):
            continue
        if curr_node is None and line[0] in ["renderer", "material", "light", "mesh", "camera", "gltf"]:
            print(f"Processing node: {line[0]}")
            curr_node = (line[0], {})
            if line[0] == "material":
                curr_node[1]["name"] = [line[1]]
        elif line[0] == "}":  # end group
            if curr_node is not None:
                nodes.append(curr_node)
            curr_node = None
        elif line[0] != "{" and curr_node is not None:
            curr_node[1][line[0]] = line[1:]
    for n in nodes:
        print(n)
    import pathlib
    scene = do_conversion(pathlib.Path(path).stem, nodes)
    with open(path[:-len(".scene")] + ".json", 'w') as file:
        json.dump(scene, file, indent=4)


if __name__ == "__main__":
    convert_scene(sys.argv[1])
