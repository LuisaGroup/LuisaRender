//
// Created by Mike Smith on 2021/12/28.
//

#pragma once

#include <rtx/ray.h>
#include <base/shape.h>
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
    Var<Shape::Handle> _shape;
    Float3 _p;
    Float3 _wo;
    Float3 _ng;
    Float2 _uv;
    Frame _shading;
    UInt _prim_id;
    Float _prim_area;
    Float _alpha;

public:
    Interaction() noexcept : _prim_id{~0u} {}
    explicit Interaction(Expr<float3> wo, Expr<float> alpha = 1.f) noexcept
        : _wo{wo}, _prim_id{~0u}, _alpha{alpha} {}
    Interaction(Expr<float3> wo, Expr<float2> uv, Expr<float> alpha = 1.f) noexcept
        : _wo{wo}, _uv{uv}, _prim_id{~0u}, _alpha{alpha} {}
    Interaction(Var<Shape::Handle> shape, Expr<uint> prim_id, Expr<float> prim_area,
                Expr<float3> p, Expr<float3> wo, Expr<float3> ng, Expr<float> alpha = 1.f) noexcept
        : _shape{std::move(shape)}, _p{p}, _wo{wo}, _ng{ng}, _shading{Frame::make(_ng)},
          _prim_id{prim_id}, _prim_area{prim_area}, _alpha{alpha} {}
    Interaction(Var<Shape::Handle> shape, Expr<uint> prim_id, Expr<float> prim_area,
                Expr<float3> p, Expr<float3> wo, Expr<float3> ng, Expr<float2> uv,
                Expr<float3> ns, Expr<float3> tangent, Expr<float> alpha = 1.f) noexcept
        : _shape{std::move(shape)}, _p{p}, _wo{wo}, _ng{ng}, _uv{uv},
          _shading{Frame::make(ite(_shape->two_sided() & (dot(ns, wo) < 0.0f), -ns, ns), tangent)},
          _prim_id{prim_id}, _prim_area{prim_area}, _alpha{alpha} {}
    [[nodiscard]] auto p() const noexcept { return _p; }
    [[nodiscard]] auto ng() const noexcept { return _ng; }
    [[nodiscard]] auto wo() const noexcept { return _wo; }
    [[nodiscard]] auto uv() const noexcept { return _uv; }
    [[nodiscard]] auto triangle_id() const noexcept { return _prim_id; }
    [[nodiscard]] auto triangle_area() const noexcept { return _prim_area; }
    [[nodiscard]] auto valid() const noexcept { return _prim_id != ~0u; }
    [[nodiscard]] auto alpha() const noexcept { return _alpha; }
    [[nodiscard]] const auto &shading() const noexcept { return _shading; }
    [[nodiscard]] const auto &shape() const noexcept { return _shape; }
    [[nodiscard]] auto wo_local() const noexcept { return _shading.world_to_local(_wo); }
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
