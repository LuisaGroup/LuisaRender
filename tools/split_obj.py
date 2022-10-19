from sys import argv
from os import path
import numpy as np


def reindex_vertices(v, iv):
    min_i, max_i = np.min(iv), np.max(iv)
    return v[min_i - 1:max_i], iv - (min_i - 1)


def process_mesh(filename, v, vt, vn, mesh):
    print(f"Processing: {filename}")
    v, iv = reindex_vertices(v, mesh[..., 0])
    vt, ivt = reindex_vertices(vt, mesh[..., 1])
    vn, ivn = reindex_vertices(vn, mesh[..., 2])
    indices = np.dstack([iv, ivt, ivn])
    with open(filename, "w") as file:
        for x in v + vt + vn:
            print(x, end="", file=file)
        print("g mesh", file=file)
        for face in indices:
            print("f", end="", file=file)
            for x, y, z in face:
                print(f" {x}/{y}/{z}", end="", file=file)
            print(file=file)


if __name__ == "__main__":
    v = []
    vn = []
    vt = []
    meshes = {}
    with open(argv[1], "r") as file:
        for i, line in enumerate(file.readlines()):
            if line.startswith("v "):
                v.append(line)
            elif line.startswith("vn "):
                vn.append(line)
            elif line.startswith("vt "):
                vt.append(line)
            elif line.startswith("usemtl ") or line.startswith("g ") or line.startswith("o "):
                name = f"Mesh.{len(meshes):05}.{'.'.join(line.split()[1:])}"
                meshes[name] = []
            elif line.startswith("f "):
                # v/vt/vn
                meshes[name].append([[int(x) for x in v.split('/')] for v in line.split()[1:]])
    for name, mesh in meshes.items():
        if mesh:
            filename = f"{path.dirname(argv[1])}/{name}.obj"
            process_mesh(filename, v, vt, vn, np.array(mesh))
