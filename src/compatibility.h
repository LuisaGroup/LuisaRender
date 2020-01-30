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
#include <atomic>

namespace metal {

using namespace glm;

using atomic_uint = std::atomic_uint;
using atomic_int = std::atomic_int;
using std::atomic_fetch_add_explicit;
using std::memory_order_relaxed;

}

using uint = uint32_t;
using uint2 = glm::uvec2;

using Vec2f = glm::vec2;

struct Vec3f : glm::vec3 {
    float padding [[maybe_unused]]{};
    using glm::vec3::vec;
    Vec3f(glm::vec3 v) noexcept : glm::vec3{v} {}
};

using Vec4f = glm::vec4;
using PackedVec2f = glm::vec2;
using PackedVec3f = glm::vec3;
using PackedVec4f = glm::vec4;

enum struct access {
    read, write, read_write
};

template<typename T, enum access method = access::read>
struct texture2d {
    Vec4f read(uint2 coord [[maybe_unused]]) { return {}; }
    void write(Vec4f color [[maybe_unused]], uint2 coord [[maybe_unused]]) {}
};

#endif

#define M_PIf        3.14159265358979323846264338327950288f   /* pi             */
#define M_PI_2f      1.57079632679489661923132169163975144f   /* pi/2           */
#define M_PI_4f      0.785398163397448309615660845819875721f  /* pi/4           */
#define M_1_PIf      0.318309886183790671537767526745028724f  /* 1/pi           */
#define M_2_PIf      0.636619772367581343075535053490057448f  /* 2/pi           */
#define M_2_SQRTPIf  1.12837916709551257389615890312154517f   /* 2/sqrt(pi)     */
#define M_SQRT2f     1.41421356237309504880168872420969808f   /* sqrt(2)        */
#define M_SQRT1_2f   0.707106781186547524400844362104849039f  /* 1/sqrt(2)      */