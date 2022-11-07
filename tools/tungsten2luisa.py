import json
from pathlib import Path
from sys import argv
import glm


def convert_roughness(r):
    return glm.sqrt(r)


def convert_albedo_texture(a):
    if isinstance(a, str):
        return f'''Image {{
    file {{ "{a}" }}
    semantic {{ "albedo" }}
  }}'''
    else:
        a = glm.vec3(a)
        return f'''Constant {{
    v {{ {a.x}, {a.y}, {a.z} }}
    semantic {{ "albedo" }}
  }}'''


def convert_emission_texture(a):
    if isinstance(a, str):
        return f'''Image {{
    file {{ "{a}" }}
    semantic {{ "illuminant" }}
  }}'''
    else:
        a = glm.vec3(a)
        return f'''Constant {{
    v {{ {a.x}, {a.y}, {a.z} }}
    semantic {{ "illuminant" }}
  }}'''


def convert_plastic_material(out_file, material: dict, alpha=""):
    name = material["name"]
    roughness = material.get("roughness", 1e-6)
    ior = material["ior"]
    color = material["albedo"]
    print(f'''
Surface mat_{name} : Substrate {{
  Kd : {convert_albedo_texture(color)}
  Ks : Constant {{
    v {{ 0.04, 0.04, 0.04 }}
    semantic {{ "albedo" }}
  }}
  eta : Constant {{
    v {{ {ior} }}
  }}
  roughness : Constant {{
    v {{ {convert_roughness(roughness)} }}
  }}{alpha}
}}''', file=out_file)


def convert_glass_material(out_file, material: dict, alpha=""):
    name = material["name"]
    color = material["albedo"]
    roughness = material.get("roughness", 1e-6)
    ior = material["ior"]
    print(f'''
Surface mat_{name} : Glass {{
  Kr : Constant {{
    v {{ 1, 1, 1 }}
    semantic {{ "albedo" }}
  }}
  Kt : {convert_albedo_texture(color)}
  eta : Constant {{
    v {{ {ior} }}
  }}
  roughness : Constant {{
    v {{ {convert_roughness(roughness)} }}
  }}{alpha}
}}''', file=out_file)


def convert_mirror_material(out_file, material: dict, alpha=""):
    name = material["name"]
    color = material["albedo"]
    print(f'''
Surface mat_{name} : Mirror {{
  color : {convert_albedo_texture(color)}{alpha}
}}''', file=out_file)


def convert_metal_material(out_file, material: dict, alpha=""):
    name = material["name"]
    if "material" in material:
        eta = f'"{material["material"]}"'
    else:
        n = material["eta"]
        k = material["k"]
        eta = f"360, {n}, {k}, 830, {n}, {k}"
    roughness = material.get("roughness", 1e-6)
    albedo = glm.vec3(material.get("albedo", 1.0))
    if albedo != glm.vec3(1):
        albedo = f"""
  Kd : Constant {{
    v {{ {albedo.x}, {albedo.y}, {albedo.z} }}
    semantic {{ "albedo" }}
  }}"""
    else:
        albedo = ""
    print(f'''
Surface mat_{name} : Metal {{
  eta {{ {eta} }}
  roughness : Constant {{
    v {{ {convert_roughness(roughness)} }}
  }}{alpha}{albedo}
}}''', file=out_file)


def convert_null_material(out_file, material: dict):
    name = material["name"]
    print(f'''
Surface mat_{name} : Null {{}}''', file=out_file)


def convert_matte_material(out_file, material: dict, alpha=""):
    name = material["name"]
    color = material["albedo"]
    print(f'''
Surface mat_{name} : Matte {{
  Kd : {convert_albedo_texture(color)}{alpha}
}}''', file=out_file)


def convert_material(out_file, material: dict, alpha=""):
    impl = material["type"]
    if impl == "plastic" or impl == "rough_plastic":
        convert_plastic_material(out_file, material, alpha)
    elif impl == "dielectric" or impl == "rough_dielectric":
        convert_glass_material(out_file, material, alpha)
    elif impl == "mirror":
        convert_mirror_material(out_file, material, alpha)
    elif impl == "conductor" or impl == "rough_conductor":
        convert_metal_material(out_file, material, alpha)
    elif impl == "lambert" or impl == "oren_nayar":
        convert_matte_material(out_file, material, alpha)
    elif impl == "transparency":  # TODO
        a = material["alpha"]
        if type(a) is float:
            aa = f'''
  alpha : Constant {{
    v {{ {float(a)} }}
  }}'''
        else:
            a_ext = Path(a).suffix
            aa = f'''
    alpha : Image {{
      file {{ "{a.replace(a_ext, f"-alpha{a_ext}")}" }}
      encoding {{ "linear" }}
  }}'''
        base_material = material["base"]
        base_material["name"] = material["name"]
        convert_material(out_file, base_material, aa)
    elif impl == "null":
        convert_null_material(out_file, material)
    else:
        print(material)
        print(f'''
Surface mat_{material["name"]} : Matte {{
  Kd : Constant {{
      v {{ 1, 1, 1 }}
      semantic {{ "albedo" }}
    }}
}}''', file=out_file)
        return
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
Env env : Spherical {{
  emission : {convert_emission_texture(emission)}
  transform : SRT {{
    rotate {{ 0, 1, 0, -90 }}
  }}
}}''', file=out_file)
    elif impl == "infinite_sphere_cap":
        power_scale = 100 * glm.pi()
        emission = glm.vec3(glm.vec3(shape["power"] / power_scale))
        angle = shape["cap_angle"]
        print(f'''
Env dir : Directional {{
  emission : Constant {{
    v {{ {emission.x}, {emission.y}, {emission.z} }}
    semantic {{ "illuminant" }}
  }}
  angle {{ {angle} }}
  transform : Matrix {{
    m {{ {M[0][0]}, {M[1][0]}, {M[2][0]}, {M[3][0]},
         {M[0][1]}, {M[1][1]}, {M[2][1]}, {M[3][1]},
         {M[0][2]}, {M[1][2]}, {M[2][2]}, {M[3][2]},
         {M[0][3]}, {M[1][3]}, {M[2][3]}, {M[3][3]} }}
  }}
  scale {{ {glm.pi() * 4.0} }}
}}''', file=out_file)
    elif impl == "skydome":
        M = glm.rotate(glm.radians(-90), glm.vec3(0, 1, 0))
        print(f'''
Env sky : Spherical {{
  emission : Image {{
    file {{ "textures/sky.exr" }}
    semantic {{ "illuminant" }}
  }}
  transform : Matrix {{
    m {{ {M[0][0]}, {M[1][0]}, {M[2][0]}, {M[3][0]},
         {M[0][1]}, {M[1][1]}, {M[2][1]}, {M[3][1]},
         {M[0][2]}, {M[1][2]}, {M[2][2]}, {M[3][2]},
         {M[0][3]}, {M[1][3]}, {M[2][3]}, {M[3][3]} }}
  }}
  scale {{ {float(shape['intensity'])} }}
}}
''', file=out_file)
        print(f"Warning: Skydome is not supported: {shape}")
    else:
        alpha = ""
        if impl == "mesh":
            file = shape["file"]
            assert file.endswith(".wo3")
            file = f"{file[:-4]}.obj"
        elif impl == "quad":
            file = "models/square.obj"
            M = M * rotateXYZ(glm.radians(glm.vec3(-90, 0, 0))
                              ) * glm.scale(glm.vec3(.5))
        elif impl == "cube":
            file = "models/cube.obj"
            M = M * rotateXYZ(glm.radians(glm.vec3(-90, 0, 0))
                              ) * glm.scale(glm.vec3(.5))
        elif impl == "disk":
            file = "models/disk.obj"
        elif impl == "sphere":
            file = "models/sphere.obj"
        elif impl == "curves":
            return
        else:
            print(f"Unsupported shape: {shape}")
            raise NotImplementedError()
        material = shape["bsdf"]
        if not isinstance(material, str):
            material = "Null"
        M0 = ", ".join(str(x) for x in glm.transpose(M)[0])
        M1 = ", ".join(str(x) for x in glm.transpose(M)[1])
        M2 = ", ".join(str(x) for x in glm.transpose(M)[2])
        M3 = ", ".join(str(x) for x in glm.transpose(M)[3])
        power_scale = 100 * glm.pi()
        emission = glm.vec3(shape.get("emission", glm.vec3(
            shape.get("power", 0)) / power_scale))
        if emission.x == emission.y == emission.z == 0:
            light = ""
        else:
            light = f'''
  light : Diffuse {{
    emission : Constant {{
      v {{ {emission.x}, {emission.y}, {emission.z} }}
      semantic {{ "illuminant" }}
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
    fov = glm.degrees(2 * glm.atan(resolution.y *
                      glm.tan(0.5 * fov) / resolution.x))
    transform = camera["transform"]
    position = glm.vec3(transform["position"])
    look_at = glm.vec3(transform["look_at"])
    front = glm.normalize(look_at - position)
    up = glm.vec3(transform["up"])
    print(f'''
Camera camera : Pinhole {{
  fov {{ {fov} }}
  spp {{ {spp} }}
  filter : Gaussian {{
    radius {{ 1 }}
  }}
  film : Color {{
    resolution {{ {int(resolution.x)}, {int(resolution.y)} }}
  }}
  file {{ "render.exr" }}
  transform : View {{
    position {{ {position.x}, {position.y}, {position.z} }}
    front {{ {front.x}, {front.y}, {front.z} }}
    up {{ {up.x}, {up.y}, {up.z} }}
  }}
}}''', file=out_file)


def write_render(out_file, shapes):
    shape_refs = ",\n    ".join(f'@shape_{i}' for i, s in enumerate(shapes)
                                if s["type"] != "infinite_sphere" and
                                s["type"] != "infinite_sphere_cap" and
                                s["type"] != "skydome")
    env = "environment : Null {}"
    if any(s["type"] == "infinite_sphere" for s in shapes):
        env = "environment { @env }"
    elif any(s["type"] == "infinite_sphere_cap" for s in shapes):
        env = "environment { @dir }"
    print(f'''
render {{
  cameras {{ @camera }}
  integrator : WavePath {{}}
  shapes {{
    {shape_refs}
  }}
  {env}
}}
''', file=out_file)


def tungsten2luisa(file_name: str, spp: int):
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


if __name__ == "__main__":
    file_name = argv[1].strip('"').strip(' ')
    spp = int(argv[2])
    tungsten2luisa(file_name, spp)
