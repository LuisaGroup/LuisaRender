from sys import argv
import json
import glm
import numpy as np


def convert_camera(file, scene):
    camera = scene["camera"]
    position = glm.vec3(camera["lookat"]["orig"])
    look_at = glm.vec3(camera["lookat"]["target"])
    front = glm.normalize(look_at - position)
    up = glm.normalize(camera["up"])
    fov = camera["fieldofview"]
    print(f"""Camera camera : Pinhole {{
  fov {{ {fov} }}
  filter : Gaussian {{
    radius {{ 1 }}
  }}
  film : Color {{}}
  spp {{ 1024 }}
  transform : View {{
    position {{ {position.x}, {position.y}, {position.z} }}
    front {{ {front.x}, {front.y}, {front.z} }}
    up {{ {up.x}, {up.y}, {up.z} }}
  }}
}}
""", file=file)


def convert_shapes(file, scene):
    for name, desc in scene["objects"].items():
        print(f'''Shape {name} : Mesh {{
  file {{ "{desc["ply"]}" }}''', file=file)
        if "transformation" in desc:
            t = np.reshape(desc["transformation"], [4, 4])
            print(f'''  transform : Matrix {{
    m {{
      {t[0][0]}, {t[1][0]}, {t[2][0]}, {t[3][0]},
      {t[0][1]}, {t[1][1]}, {t[2][1]}, {t[3][1]},
      {t[0][2]}, {t[1][2]}, {t[2][2]}, {t[3][2]},
      {t[0][3]}, {t[1][3]}, {t[2][3]}, {t[3][3]}
    }}
  }}''', file=file)
        print(f'''  surface {{ @{desc["material"]} }}''', file=file)
        if desc["material"] in scene["lights"]:
            print(f"  light {{ @{desc['material']}_EMISSION }}", file=file)
        print("}\n", file=file)


def convert_surfaces(file, scene):
    for name, desc in scene["materials"].items():
        transparency = ""
        if "transparency" in desc:
            transparency = f'''
  alpha {{ @{desc["transparency"]} }}'''
        if desc["type"] == "matte":
            print(f'''Surface {name} : Matte {{
  Kd {{ @{desc["kd"]} }}{transparency}
}}
''', file=file)
        elif desc["type"] == "roughmatte":
            print(f'''Surface {name} : Matte {{
  Kd {{ @{desc["kd"]} }}
  sigma {{ @{desc["sigma"]} }}{transparency}
}}
''', file=file)
        elif desc["type"] == "glass":
            print(f'''Surface {name} : Glass {{
  Kr {{ @{desc["kr"]} }}
  Kt {{ @{desc["kt"]} }}{transparency}
  roughness : Constant {{
    v {{ 0.2 }}
  }}
}}
''', file=file)
        else:
            raise NotImplementedError(f"Surface {name} : {desc['type']}")


def convert_lights(file, scene):
    for name, desc in scene["lights"].items():
        if isinstance(desc["emission"], str):
            print(f'''Light {name}_EMISSION : Diffuse {{
  emission {{ @{desc["emission"]} }}
  scale {{ 100 }}
}}
''', file=file)
        elif desc["emission"]["power"] != 0:
            raise NotImplementedError()


def convert_textures(file, scene):
    for name, desc in scene["textures"].items():
        if desc["type"] == "imagemap":
            f = desc["file"]
            s = desc["gain"]
            g = desc["gamma"]
            uv_scale = glm.vec2(desc["mapping"]["uvscale"])
            uv_offset = glm.vec2(desc["mapping"]["uvdelta"])
            print(f'''Texture {name} : Image {{
  file {{ "{f}" }}
  encoding {{ "gamma" }}
  gamma {{ {g} }}
  scale {{ {s} }}
  uv_scale {{ {uv_scale.x}, {-uv_scale.y} }}
  uv_offset {{ {uv_offset.x}, {uv_offset.y} }}
}}
''', file=file)
        elif desc["type"] == "constfloat1":
            print(f'''Texture {name} : Constant {{
  v {{ {desc["value"]} }}
}}
''', file=file)
        elif desc["type"] == "constfloat2":
            print(f'''Texture {name} : Constant {{
  v {{ {desc["value"][0]}, {desc["value"][1]} }}
}}
''', file=file)
        elif desc["type"] == "constfloat3":
            print(f'''Texture {name} : Constant {{
  v {{ {desc["value"][0]}, {desc["value"][1]}, {desc["value"][2]} }}
}}
''', file=file)
        elif desc["type"] == "constfloat4":
            print(f'''Texture {name} : Constant {{
  v {{ {desc["value"][0]}, {desc["value"][1]}, {desc["value"][2]}, {desc["value"][3]} }}
}}
''', file=file)
        else:
            raise NotImplementedError()


def lux2luisa(file_name: str):
    assert file_name.endswith(".scn")
    with open(file_name) as file:
        lines = [line.split("=") for line in file.readlines()]
    result = {}

    def process_property(key: str, value: str):
        key = key.strip().split(".")
        value = value.strip()
        if value.startswith('"'):
            value = value.strip('"')
        else:
            values = value.split()
            if len(values) == 1:
                value = int(float(value)) if int(float(value)) == float(value) else float(value)
            else:
                value = [float(v) for v in values]
        parent = result
        for seg in key[:-1]:
            if seg not in parent:
                parent[seg] = {}
            parent = parent[seg]
        parent[key[-1]] = value

    for line in lines:
        process_property(*line)
    assert len(result) == 1 and "scene" in result
    scene = result["scene"]
    scene["lights"] = {}
    for mat, desc in scene["materials"].items():
        if isinstance(desc["emission"], str) or desc["emission"]["power"] != 0:
            scene["lights"][mat] = desc
    with open(f"{file_name[:-4]}.json", "w") as file:
        json.dump(scene, file, indent=2)
    with open(f"{file_name[:-4]}.luisa", "w") as file:
        convert_camera(file, scene)
        convert_textures(file, scene)
        convert_surfaces(file, scene)
        convert_lights(file, scene)
        convert_shapes(file, scene)
        shapes_str = ",\n    @".join(scene["objects"])
        print(f'''render {{
  cameras {{ @camera }}
  integrator : WavePath {{
    spectrum : Hero {{}}
    depth {{ 12 }}
    rr_depth {{ 4 }}
  }}
  shapes {{
    @{shapes_str}
  }}
}}''', file=file)


if __name__ == "__main__":
    file_name = argv[1].strip('"').strip(' ')
    lux2luisa(file_name)
