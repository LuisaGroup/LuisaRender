//
// Created by Mike on 8/28/2020.
//

#pragma once

#include <core/context.h>
#include <map>

namespace luisa::cuda {

constexpr auto texture_jit_header =
    "#pragma once\n"
    "\n"
    "#include <type_traits>\n"
    "#include <math_util.h>\n"
    "\n"
    "namespace luisa {\n"
    "\n"
    "template<typename T>\n"
    "struct Tex2D {\n"
    "    cudaTextureObject_t texture;\n"
    "    cudaSurfaceObject_t surface;\n"
    "};\n"
    "template<typename U, typename V> struct IsSameImpl { static constexpr auto value = false; };\n"
    "template<typename U> struct IsSameImpl<U, U> { static constexpr auto value = true; };\n"
    "template<typename U, typename V> constexpr auto is_same_v = IsSameImpl<U, V>::value;\n"
    "\n"
    "template<typename T>\n"
    "inline float4 sample(Tex2D<T> t, luisa::float2 coord) noexcept {\n"
    "    auto v = tex2D<::float4>(t.texture, coord.x, coord.y);\n"
    "    return make_float4(v.x, v.y, v.z, v.w);\n"
    "}\n"
    "\n"
    "template<typename T>\n"
    "inline float4 read(Tex2D<T> t, luisa::uint2 coord) noexcept {\n"
    "    constexpr auto denom = 1.0f / 255.0f;\n"
    "    if constexpr (is_same_v<T, float>) {\n"
    "        auto pixel = surf2Dread<float>(t.surface, coord.x * 4u, coord.y);\n"
    "        return make_float4(pixel, 0.0f, 0.0f, 0.0f);\n"
    "    } else if constexpr (is_same_v<T, float2>) {\n"
    "        auto pixel = surf2Dread<::float2>(t.surface, coord.x * 8u, coord.y);\n"
    "        return make_float4(pixel.x, pixel.y, 0.0f, 0.0f);\n"
    "    } else if constexpr (is_same_v<T, float4>) {\n"
    "        auto pixel = surf2Dread<::float4>(t.surface, coord.x * 16u, coord.y);\n"
    "        return make_float4(pixel.x, pixel.y, pixel.z, pixel.w);\n"
    "    } else if constexpr (is_same_v<T, uint8_t>) {\n"
    "        auto pixel = surf2Dread<uint8_t>(t.surface, coord.x, coord.y);\n"
    "        return make_float4(pixel * denom, 0.0f, 0.0f, 0.0f);\n"
    "    } else if constexpr (is_same_v<T, uchar2>) {\n"
    "        auto pixel = surf2Dread<::uchar2>(t.surface, coord.x * 2u, coord.y);\n"
    "        return make_float4(pixel.x * denom, pixel.y * denom, 0.0f, 0.0f);\n"
    "    } else if constexpr (is_same_v<T, uchar4>) {\n"
    "        auto pixel = surf2Dread<::uchar4>(t.surface, coord.x * 4u, coord.y);\n"
    "        return make_float4(pixel.x * denom, pixel.y * denom, pixel.z * denom, pixel.w * denom);\n"
    "    }\n"
    "    return make_float4(0.0f);\n"
    "}\n"
    "\n"
    "template<typename T>\n"
    "inline void write(Tex2D<T> t, luisa::uint2 coord, luisa::float4 pixel) noexcept {\n"
    "    if constexpr (is_same_v<T, float>) {\n"
    "        surf2Dwrite<float>(pixel.x, t.surface, coord.x * 4u, coord.y);\n"
    "    } else if constexpr (is_same_v<T, float2>) {\n"
    "        auto p = ::float2{pixel.x, pixel.y};\n"
    "        surf2Dwrite<::float2>(p, t.surface, coord.x * 8u, coord.y);\n"
    "    } else if constexpr (is_same_v<T, float4>) {\n"
    "        auto p = ::float4{pixel.x, pixel.y, pixel.z, pixel.w};\n"
    "        surf2Dwrite<::float4>(p, t.surface, coord.x * 16u, coord.y);\n"
    "    } else if constexpr (is_same_v<T, uint8_t>) {\n"
    "        auto p = static_cast<uint8_t>(clamp(round(pixel.x * 255.0f), 0.0f, 255.0f));\n"
    "        surf2Dwrite<uint8_t>(p, t.surface, coord.x, coord.y);\n"
    "    } else if constexpr (is_same_v<T, uchar2>) {\n"
    "        auto v = clamp(round(pixel * 255.0f), 0.0f, 255.0f);\n"
    "        auto p = ::uchar2{\n"
    "            static_cast<uint8_t>(v.x),\n"
    "            static_cast<uint8_t>(v.y)};\n"
    "        surf2Dwrite<::uchar2>(p, t.surface, coord.x * 2u, coord.y);\n"
    "    } else if constexpr (is_same_v<T, uchar4>) {\n"
"            auto v = clamp(round(pixel * 255.0f), 0.0f, 255.0f);\n"
    "        auto p = ::uchar4{\n"
    "            static_cast<uint8_t>(v.x),\n"
    "            static_cast<uint8_t>(v.y),\n"
    "            static_cast<uint8_t>(v.z),\n"
    "            static_cast<uint8_t>(v.w)};\n"
    "        surf2Dwrite<::uchar4>(p, t.surface, coord.x * 4u, coord.y);\n"
    "    }\n"
    "}\n"
    "\n"
    "}\n";

inline const auto &get_jit_headers(Context *context) noexcept {
    static std::map<const char *, std::string> headers{
        {"scalar_types.h", text_file_contents(context->runtime_path("include") / "core" / "scalar_types.h")},
        {"vector_types.h", text_file_contents(context->runtime_path("include") / "core" / "vector_types.h")},
        {"matrix_types.h", text_file_contents(context->runtime_path("include") / "core" / "matrix_types.h")},
        {"data_types.h", text_file_contents(context->runtime_path("include") / "core" / "data_types.h")},
        {"math_util.h", text_file_contents(context->runtime_path("include") / "core" / "math_helpers.h")},
        {"texture_util.h", texture_jit_header}};
    return headers;
}

;}// namespace luisa::cuda
