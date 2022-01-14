//
// Created by Mike Smith on 2022/1/13.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

using luisa::compute::Expr;
using luisa::compute::Float3;

class Frame {

private:
    Float3 _u;
    Float3 _v;
    Float3 _n;

private:
    Frame(Float3 tangent, Float3 bitangent, Float3 normal) noexcept;

public:
    Frame() noexcept;
    [[nodiscard]] static Frame make(Expr<float3> normal) noexcept;
    [[nodiscard]] static Frame make(Expr<float3> normal, Expr<float3> tangent) noexcept;
    [[nodiscard]] Float3 local_to_world(Expr<float3> d) const noexcept;
    [[nodiscard]] Float3 world_to_local(Expr<float3> d) const noexcept;
    [[nodiscard]] Expr<float3> u() const noexcept { return _u; }
    [[nodiscard]] Expr<float3> v() const noexcept { return _v; }
    [[nodiscard]] Expr<float3> n() const noexcept { return _n; }
};


}
