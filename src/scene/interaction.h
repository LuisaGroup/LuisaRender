//
// Created by Mike Smith on 2021/12/28.
//

#pragma once

#include <rtx/ray.h>
#include <scene/shape.h>
#include <util/frame.h>

namespace luisa::render {

using luisa::compute::Bool;
using luisa::compute::Expr;
using luisa::compute::Float2;
using luisa::compute::Float3;
using luisa::compute::Float4x4;
using luisa::compute::Ray;
using luisa::compute::UInt;

class Interaction {

private:
    Var<InstancedShape> _shape;
    Float3 _p;
    Float3 _wo;
    Float3 _ng;
    Float2 _uv;
    Frame _shading;
    UInt _prim_id;
    Float _prim_area;

public:
    Interaction() noexcept : _prim_id{~0u} {}
    explicit Interaction(Expr<float3> wo) noexcept : _wo{wo}, _prim_id{~0u} {}
    Interaction(Var<InstancedShape> shape, Expr<uint> prim_id, Expr<float> prim_area,
                Expr<float3> p, Expr<float3> wo, Expr<float3> ng) noexcept
        : _shape{std::move(shape)}, _p{p}, _wo{wo},
          _ng{ng}, _shading{Frame::make(_ng)},
          _prim_id{prim_id}, _prim_area{prim_area} {}
    Interaction(Var<InstancedShape> shape, Expr<uint> prim_id, Expr<float> prim_area,
                Expr<float3> p, Expr<float3> wo, Expr<float3> ng,
                Expr<float2> uv, Expr<float3> ns, Expr<float3> tangent) noexcept
        : _shape{std::move(shape)}, _p{p}, _wo{wo}, _ng{ng}, _uv{uv},
          _shading{Frame::make(ite(_shape->two_sided() & (dot(ns, wo) < 0.0f), -ns, ns), tangent)},
          _prim_id{prim_id}, _prim_area{prim_area} {}
    [[nodiscard]] auto p() const noexcept { return _p; }
    [[nodiscard]] auto ng() const noexcept { return _ng; }
    [[nodiscard]] auto wo() const noexcept { return _wo; }
    [[nodiscard]] auto uv() const noexcept { return _uv; }
    [[nodiscard]] auto triangle_id() const noexcept { return _prim_id; }
    [[nodiscard]] auto triangle_area() const noexcept { return _prim_area; }
    [[nodiscard]] auto valid() const noexcept { return _prim_id != ~0u; }
    [[nodiscard]] const auto &shading() const noexcept { return _shading; }
    [[nodiscard]] const auto &shape() const noexcept { return _shape; }
    [[nodiscard]] auto spawn_ray(Expr<float3> wi) const noexcept {
        return luisa::compute::make_ray_robust(_p, _ng, wi);
    }
    [[nodiscard]] auto spawn_ray_to(Expr<float3> p_light, Expr<float> eps = 1e-3f) const noexcept {
        using namespace luisa::compute;
        auto l = p_light - _p;
        return make_ray_robust(_p, _ng, normalize(l), length(l) - eps);
    }
};

}// namespace luisa::render
