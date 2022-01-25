//
// Created by Mike Smith on 2022/1/19.
//

#include <util/colorspace.h>

namespace luisa::render {

Float3 cie_xyz_to_linear_srgb(Expr<float3> xyz) noexcept {
    constexpr auto m = make_float3x3(
        +3.240479f, -0.969256f, +0.055648f,
        -1.537150f, +1.875991f, -0.204043f,
        -0.498535f, +0.041556f, +1.057311f);
    return m * xyz;
}

}// namespace luisa::render
