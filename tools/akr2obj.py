from struct import unpack_from, calcsize
from sys import argv
import numpy as np


# returns (positions, normals, texcoords, position_indices, normal_indices, texcoord_indices)
def decode_akari_mesh(buffer):
    def decode(fmt, offset):
        size = calcsize(fmt)
        data = unpack_from(fmt, buffer, offset)
        return data, offset + size

    # [len] [name (byte)  ]
    # [len] [v    (float3)]
    # [len] [vn   (float3)]
    # [len] [vt   (float2)]
    # [len] [iv   (uint3) ]
    # [len] [ivn  (uint3) ]
    # [len] [ivt  (uint3) ]
    name_len, offset = decode("Q", 0)
    name, offset = decode(f"{name_len[0]}s", offset)
    print(f"Processing '{str(name[0].decode('utf-8'))}'")
    v_len, offset = decode("Q", offset)
    v, offset = decode(f"{v_len[0] * 3}f", offset)
    vn_len, offset = decode("Q", offset)
    vn, offset = decode(f"{vn_len[0] * 3}f", offset)
    vt_len, offset = decode("Q", offset)
    vt, offset = decode(f"{vt_len[0] * 2}f", offset)
    iv_len, offset = decode("Q", offset)
    iv, offset = decode(f"{iv_len[0] * 3}I", offset)
    ivn_len, offset = decode("Q", offset)
    ivn, offset = decode(f"{ivn_len[0] * 3}I", offset)
    ivt_len, offset = decode("Q", offset)
    ivt, offset = decode(f"{ivt_len[0] * 3}I", offset)
    return np.reshape(v, [-1, 3]), \
           np.reshape(vn, [-1, 3]), \
           np.reshape(vt, [-1, 2]), \
           np.reshape(iv, [-1, 3]), \
           np.reshape(ivn, [-1, 3]), \
           np.reshape(ivt, [-1, 3])


if __name__ == "__main__":
    filename = argv[1]
    assert filename.endswith(".mesh")
    with open(filename, "rb") as file:
        [p, n, t, pi, ni, ti] = decode_akari_mesh(file.read())
    assert pi.shape == ni.shape == ti.shape
    print(f"vertices: {p.shape}, triangles: {pi.shape}")
    with open(f"{filename[:-5]}.obj", "w") as file:
        for px, py, pz in p:
            print(f"v {px} {py} {pz}", file=file)
        for nx, ny, nz in n:
            print(f"vn {nx} {ny} {nz}", file=file)
        for tx, ty in t:
            print(f"vt {tx} {ty}", file=file)
        for (pix, piy, piz), (nix, niy, niz), (tix, tiy, tiz) in zip(pi + 1, ni + 1, ti + 1):
            print(f"f {pix}/{tix}/{nix} {piy}/{tiy}/{niy} {piz}/{tiz}/{niz}", file=file)
