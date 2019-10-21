//
// Created by Mike Smith on 2019/10/20.
//

#pragma once

#ifdef __METAL_VERSION__

#include <metal_stdlib>
using Vec2f = float2;
using Vec3f = float3;
using Vec4f = float4;
using PackedVec2f = packed_float2;
using PackedVec3f = packed_float3;
using PackedVec4f = packed_float4;

#else

#include <glm/glm.hpp>

inline namespace metal {

using uint = uint32_t;
using uint2 = glm::uvec2;

using Vec2f = glm::vec2;

struct Vec3f : glm::vec3 {
    float padding [[maybe_unused]];
    using glm::vec3::vec;
};

using Vec4f = glm::vec4;
using PackedVec2f = glm::vec2;
using PackedVec3f = glm::vec3;
using PackedVec4f = glm::vec4;

using glm::normalize;

}

#define constant const
#define device
#define thread
#define kernel

#endif