import json
from sys import argv
from glm import *


def convert_plastic_material(out_file, material: dict):
    name = material["name"]
    roughness = material.get("roughness", 0.01)
    ior = material["ior"]
    color = vec3(material["albedo"])
    print(f'''
Surface mat_{name} : Substrate {{
  Kd : ConstColor {{
    color {{ {color.x}, {color.y}, {color.z} }}
  }}
  Ks : ConstColor {{
    color {{ 0.4 }}
  }}
  eta : ConstGeneric {{
    v {{ {ior} }}
  }}
  roughness : ConstGeneric {{
    v {{ {sqrt(roughness)} }}
  }}
}}''', file=out_file)


def convert_glass_material(out_file, material: dict):
    name = material["name"]
    color = vec3(material["albedo"])
    ior = material["ior"]
    print(f'''
Surface mat_{name} : Glass {{
  Kr : ConstColor {{
    color {{ 1 }}
  }}
  Kt : ConstColor {{
    color {{ {color.x}, {color.y}, {color.z} }}
  }}
  eta : ConstGeneric {{
    v {{ {ior} }}
  }}
  roughness : ConstGeneric {{
    v {{ 0.01 }}
  }}
}}''', file=out_file)


def convert_mirror_material(out_file, material: dict):
    name = material["name"]
    color = vec3(material["albedo"])
    print(f'''
Surface mat_{name} : Mirror {{
  color : ConstColor {{
    color {{ {color.x}, {color.y}, {color.z} }}
  }}
}}''', file=out_file)


def convert_material(out_file, material: dict):
    impl = material["type"]
    if impl == "plastic" or impl == "rough_plastic":
        convert_plastic_material(out_file, material)
    elif impl == "dielectric":
        convert_glass_material(out_file, material)
    elif impl == "mirror":
        convert_mirror_material(out_file, material)
    else:
        raise NotImplemented


def convert_materials(out_file, materials):
    for mat in materials:
        convert_material(out_file, mat)
    print("\nSurface mat_Null : Null {}", file=out_file)


def rotateYXZ(r):
    c = cos(r)
    s = sin(r)
    return mat4(
        c[1] * c[2] - s[1] * s[0] * s[2], -c[1] * s[2] - s[1] * s[0] * c[2], -s[1] * c[0], 0.0,
        c[0] * s[2], c[0] * c[2], -s[0], 0.0,
        s[1] * c[2] + c[1] * s[0] * s[2], -s[1] * s[2] + c[1] * s[0] * c[2], c[1] * c[0], 0.0,
        0.0, 0.0, 0.0, 1.0)


def convert_transform(S, R, T):
    x = vec3(1.0, 0.0, 0.0) * S.x
    y = vec3(0.0, 1.0, 0.0) * S.y
    z = vec3(0.0, 0.0, 1.0) * S.z
    R = transpose(rotateYXZ(R))
    x = vec3(R * vec4(x, 1))
    y = vec3(R * vec4(y, 1))
    z = vec3(R * vec4(z, 1))
    return mat4(
        x[0], y[0], z[0], T[0],
        x[1], y[1], z[1], T[1],
        x[2], y[2], z[2], T[2],
        0.0, 0.0, 0.0, 1.0)


def convert_shape(out_file, index, shape: dict):
    transform = shape["transform"]
    T = vec3(transform.get("position", 0))
    R = radians(vec3(transform.get("rotation", 0)))
    S = vec3(transform.get("scale", 1))
    M = transpose(convert_transform(S, R, T))
    M0 = ", ".join(str(x) for x in transpose(M)[0])
    M1 = ", ".join(str(x) for x in transpose(M)[1])
    M2 = ", ".join(str(x) for x in transpose(M)[2])
    M3 = ", ".join(str(x) for x in transpose(M)[3])
    impl = shape["type"]
    material = shape["bsdf"]
    if not isinstance(material, str):
        material = "Null"
    if impl == "mesh":
        file = shape["file"]
        assert file.endswith(".wo3")
        file = f"{file[:-4]}.obj"
    elif impl == "quad":
        file = "models/square.obj"
    else:
        raise NotImplemented
    emission = vec3(shape.get("emission", 0))
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
  surface {{ @mat_{material} }}{light}
  transform : Matrix {{
    m {{
      {M0},
      {M1},
      {M2},
      {M3}
    }}
  }}
}}''', file=out_file)


def convert_shapes(out_file, shapes):
    for i, shape in enumerate(shapes):
        convert_shape(out_file, i, shape)


def convert_camera(out_file, camera: dict, spp):
    resolution = vec2(camera["resolution"])
    fov = camera["fov"]
    transform = camera["transform"]
    position = vec3(transform["position"])
    look_at = vec3(transform["look_at"])
    front = normalize(look_at - position)
    up = vec3(transform["up"])
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
    file {{ "color.exr" }}
    resolution {{ {resolution.x}, {resolution.y} }}
  }}
}}''', file=out_file)


def write_render(out_file, shapes):
    shape_refs = ",\n    ".join(f'@shape_{i}' for i in range(len(shapes)))
    print(f'''
render {{
  cameras {{ @camera }}
  integrator : MegaPath {{}}
  shapes {{
    {shape_refs}
  }}
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
        convert_shapes(file, shapes)
        convert_camera(file, camera, spp)
        write_render(file, shapes)
