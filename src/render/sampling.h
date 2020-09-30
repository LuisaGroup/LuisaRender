//
// Created by Mike Smith on 2020/9/16.
//

#pragma once

#include <compute/dsl_syntax.h>

namespace luisa::render {

inline namespace sampling {

struct DiscreteSample {
    uint index;
    float pdf;
};

}
}

LUISA_STRUCT(luisa::render::sampling::DiscreteSample, index, pdf)

namespace luisa::render {

inline namespace sampling {

using compute::dsl::Var;
using compute::dsl::Expr;

inline Expr<float3> uniform_sample_hemisphere(Var<float2> u) {
    Var r = sqrt(1.0f - u.x * u.x);
    Var phi = 2.0f * constants::pi * u.y;
    return make_float3(r * cos(phi), r * sin(phi), u.x);
}

inline Expr<float3> cosine_sample_hemisphere(Var<float2> u) {
    Var r = sqrt(u.x);
    Var phi = 2.0f * constants::pi * u.y;
    return make_float3(r * cos(phi), r * sin(phi), sqrt(1.0f - u.x));
}

inline Expr<float2> uniform_sample_triangle(Var<float2> u) {
    Var sx = sqrt(u.x);
    return make_float2(1.0f - sx, u.y * sx);
}

template<typename Table, typename = decltype(std::declval<Table &>()[Expr{0u}])>
inline Expr<DiscreteSample> sample_discrete(Table &&cdf, Var<uint> first, Var<uint> last, Var<float> u) noexcept {
    Var first_copy = first;
    Var count = last - first;
    While (count > 0u) {
        Var step = count / 2u;
        Var it = first + step;
        Var pred = cdf[it] < u;
        first = select(pred, it + 1u, first);
        count = select(pred, count - (step + 1u), step);
    };
    Var<DiscreteSample> sample;
    sample.index = min(first, last - 1u);
    If (sample.index == first_copy) {
        sample.pdf = cdf[sample.index];
    } Else {
        sample.pdf = cdf[sample.index] - cdf[sample.index - 1u];
    };
    return sample;
}

}

}
