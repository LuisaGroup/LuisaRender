//
// Created by Mike Smith on 2022/1/19.
//

#include <util/colorspace.h>

namespace luisa::render {

// from Wikipedia: https://en.wikipedia.org/wiki/SRGB#The_sRGB_transfer_function_.28.22gamma.22.29
Float3 cie_xyz_to_linear_srgb(Expr<float3> xyz) noexcept {
    auto m = make_float3x3(
        +3.2406255f, -0.9689307f, +0.0557101f,
        -1.5372080f, +1.8757561f, -0.2040211f,
        -0.4986286f, +0.0415175f, +1.0569959f);
    return m * xyz;
}

}// namespace luisa::render
