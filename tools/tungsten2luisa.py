import json
from pathlib import Path
from sys import argv
import glm


def convert_roughness(r):
    return glm.sqrt(r)


def convert_albedo_texture(a):
    if isinstance(a, str):
        return f'''Color {{
    file {{ "{a}" }}
    uv_scale {{ 1, -1 }}
  }}'''
    else:
        a = glm.vec3(a)
        return f'''ConstColor {{
    color {{ {a.x}, {a.y}, {a.z} }}
  }}'''


def convert_emission_texture(a):
    if isinstance(a, str):
        return f'''Illum {{
    file {{ "{a}" }}
  }}'''
    else:
        a = glm.vec3(a)
        return f'''ConstIllum {{
    color {{ {a.x}, {a.y}, {a.z} }}
  }}'''


def convert_plastic_material(out_file, material: dict):
    name = material["name"]
    roughness = material.get("roughness", 1e-6)
    ior = material["ior"]
    color = material["albedo"]
    print(f'''
Surface mat_{name} : Substrate {{
  Kd : {convert_albedo_texture(color)}
  Ks : ConstColor {{
    color {{ 0.04 }}
  }}
  eta : ConstGeneric {{
    v {{ {ior} }}
  }}
  roughness : ConstGeneric {{
    v {{ {convert_roughness(roughness)} }}
  }}
}}''', file=out_file)


def convert_glass_material(out_file, material: dict):
    name = material["name"]
    color = material["albedo"]
    roughness = material.get("roughness", 1e-6)
    ior = material["ior"]
    print(f'''
Surface mat_{name} : Glass {{
  Kr : ConstColor {{
    color {{ 1 }}
  }}
  Kt : {convert_albedo_texture(color)}
  eta : ConstGeneric {{
    v {{ {ior} }}
  }}
  roughness : ConstGeneric {{
    v {{ {convert_roughness(roughness)} }}
  }}
}}''', file=out_file)


def convert_mirror_material(out_file, material: dict):
    name = material["name"]
    color = material["albedo"]
    print(f'''
Surface mat_{name} : Mirror {{
  color : {convert_albedo_texture(color)}
}}''', file=out_file)


def convert_metal_material(out_file, material: dict):
    name = material["name"]
    if "material" in material:
        eta = f'"{material["material"]}"'
    else:
        n = material["eta"]
        k = material["k"]
        eta = f"360, {n}, {k}, 830, {n}, {k}"
    roughness = material.get("roughness", 1e-6)
    print(f'''
Surface mat_{name} : Metal {{
  eta {{ {eta} }}
  roughness : ConstGeneric {{
    v {{ {convert_roughness(roughness)} }}
  }}
}}''', file=out_file)


def convert_null_material(out_file, material: dict):
    name = material["name"]
    print(f'''
Surface mat_{name} : Null {{}}''', file=out_file)


def convert_matte_material(out_file, material: dict):
    name = material["name"]
    color = material["albedo"]
    print(f'''
Surface mat_{name} : Matte {{
  Kd : {convert_albedo_texture(color)}
}}''', file=out_file)


def convert_material(out_file, material: dict):
    impl = material["type"]
    if impl == "plastic" or impl == "rough_plastic":
        convert_plastic_material(out_file, material)
    elif impl == "dielectric" or impl == "rough_dielectric":
        convert_glass_material(out_file, material)
    elif impl == "mirror":
        convert_mirror_material(out_file, material)
    elif impl == "conductor" or impl == "rough_conductor":
        convert_metal_material(out_file, material)
    elif impl == "lambert" or impl == "oren_nayar":
        convert_matte_material(out_file, material)
    elif impl == "transparency":  # TODO
        base_material = material["base"]
        base_material["name"] = material["name"]
        convert_material(out_file, base_material)
    elif impl == "null":
        convert_null_material(out_file, material)
    else:
        print(material)
        raise NotImplementedError()


def convert_materials(out_file, materials):
    for mat in materials:
        convert_material(out_file, mat)
    print("\nSurface mat_Null : Null {}", file=out_file)


def rotateXYZ(R):
    return glm.rotate(R.z, (0, 0, 1)) * glm.rotate(R.y, (0, 1, 0)) * glm.rotate(R.x, (1, 0, 0))


def rotateYXZ(R):
    return glm.rotate(R.z, (0, 0, 1)) * glm.rotate(R.x, (1, 0, 0)) * glm.rotate(R.y, (0, 1, 0))


def convert_transform(S, R, T):
    return glm.translate(T) * rotateYXZ(R) * glm.scale(S)


def convert_shape(out_file, index, shape: dict, materials: dict):
    transform = shape["transform"]
    T = glm.vec3(transform.get("position", 0))
    R = glm.radians(glm.vec3(transform.get("rotation", 0)))
    S = glm.vec3(transform.get("scale", 1))
    M = convert_transform(S, R, T)
    impl = shape["type"]
    if impl == "infinite_sphere":
        emission = shape["emission"]
        print(f'''
Env env : Map {{
  emission : {convert_emission_texture(emission)}
  transform : SRT {{
    rotate {{ 0, 1, 0, -90 }}
  }}
}}''', file=out_file)
    else:
        alpha = ""
        if impl == "mesh":
            file = shape["file"]
            assert file.endswith(".wo3")
            file = f"{file[:-4]}.obj"
            if "bsdf" in shape:
                mat_name = shape["bsdf"]
                for m in materials:
                    if m["name"] == mat_name and m["type"] == "transparency":
                        a = m["alpha"]
                        a_ext = Path(a).suffix
                        alpha = f'''
  alpha {{ "{a.replace(a_ext, f"-alpha{a_ext}")}" }}'''
                        break
        elif impl == "quad":
            file = "models/square.obj"
            M = M * rotateXYZ(glm.radians(glm.vec3(-90, 0, 0))) * glm.scale(glm.vec3(.5))
        elif impl == "cube":
            file = "models/cube.obj"
            M = M * rotateXYZ(glm.radians(glm.vec3(-90, 0, 0))) * glm.scale(glm.vec3(.5))
        else:
            print(shape)
            raise NotImplementedError()
        material = shape["bsdf"]
        if not isinstance(material, str):
            material = "Null"
        M0 = ", ".join(str(x) for x in glm.transpose(M)[0])
        M1 = ", ".join(str(x) for x in glm.transpose(M)[1])
        M2 = ", ".join(str(x) for x in glm.transpose(M)[2])
        M3 = ", ".join(str(x) for x in glm.transpose(M)[3])
        power_scale = 100 * glm.pi()
        emission = glm.vec3(shape.get("emission", glm.vec3(shape.get("power", 0)) / power_scale))
        if emission.x == emission.y == emission.z == 0:
            light = ""
        else:
            light = f'''
  light : Diffuse {{
    emission : ConstIllum {{
      emission {{ {emission.x}, {emission.y}, {emission.z} }}
    }}
  }}'''
        print(f'''
Shape shape_{index} : Mesh {{
  file {{ "{file}" }}
  surface {{ @mat_{material} }}{light}{alpha}
  transform : Matrix {{
    m {{
      {M0},
      {M1},
      {M2},
      {M3}
    }}
  }}
}}''', file=out_file)


def convert_shapes(out_file, shapes, materials):
    for i, shape in enumerate(shapes):
        convert_shape(out_file, i, shape, materials)


def convert_camera(out_file, camera: dict, spp):
    resolution = glm.vec2(camera["resolution"])
    fov = glm.radians(camera["fov"])
    fov = glm.degrees(2 * glm.atan(resolution.y * glm.tan(0.5 * fov) / resolution.x))
    transform = camera["transform"]
    position = glm.vec3(transform["position"])
    look_at = glm.vec3(transform["look_at"])
    front = glm.normalize(look_at - position)
    up = glm.vec3(transform["up"])
    print(f'''
Camera camera : Pinhole {{
  position {{ {position.x}, {position.y}, {position.z} }}
  front {{ {front.x}, {front.y}, {front.z} }}
  up {{ {up.x}, {up.y}, {up.z} }}
  fov {{ {fov} }}
  spp {{ {spp} }}
  filter : Gaussian {{
    radius {{ 1 }}
  }}
  film : Color {{
    resolution {{ {int(resolution.x)}, {int(resolution.y)} }}
  }}
  file {{ "color.exr" }}
}}''', file=out_file)


def write_render(out_file, shapes):
    shape_refs = ",\n    ".join(f'@shape_{i}' for i, s in enumerate(shapes) if s["type"] != "infinite_sphere")
    env = "environment { @env }" if any(
        s["type"] == "infinite_sphere" for s in shapes) else "environment : Null {}"
    print(f'''
render {{
  cameras {{ @camera }}
  integrator : MegaPath {{}}
  shapes {{
    {shape_refs}
  }}
  {env}
}}
''', file=out_file)


if __name__ == "__main__":
    file_name = argv[1]
    spp = int(argv[2])
    assert file_name.endswith(".json")
    with open(file_name) as file:
        scene = json.load(file)
    materials = scene["bsdfs"]
    shapes = scene["primitives"]
    camera = scene["camera"]
    print(materials)
    print(shapes)
    print(camera)
    with open(f"{file_name[:-5]}.luisa", "w") as file:
        convert_materials(file, materials)
        convert_shapes(file, shapes, materials)
        convert_camera(file, camera, spp)
        write_render(file, shapes)
