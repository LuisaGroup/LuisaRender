//
// Created by Mike Smith on 2021/12/28.
//

#pragma once

#include <rtx/ray.h>
#include <scene/shape.h>

namespace luisa::render {

using luisa::compute::Bool;
using luisa::compute::Expr;
using luisa::compute::Float2;
using luisa::compute::Float3;
using luisa::compute::Float4x4;
using luisa::compute::Ray;
using luisa::compute::UInt;

class Frame {

private:
    Float3 _u;
    Float3 _v;
    Float3 _n;

private:
    Frame(Float3 tangent, Float3 bitangent, Float3 normal) noexcept
        : _u{std::move(tangent)},
          _v{std::move(bitangent)},
          _n{std::move(normal)} {}

public:
    Frame() noexcept
        : _u{1.0f, 0.0f, 0.0f},
          _v{0.0f, 1.0f, 0.0f},
          _n{0.0f, 0.0f, 1.0f} {}
    [[nodiscard]] static auto make(Expr<float3> normal) noexcept {
        auto bitangent = normalize(ite(
            abs(normal.x) > abs(normal.z),
            make_float3(-normal.y, normal.x, 0.0f),
            make_float3(0.0f, -normal.z, normal.y)));
        auto tangent = normalize(cross(bitangent, normal));
        return Frame{std::move(tangent), std::move(bitangent), normal};
    }
    [[nodiscard]] static auto make(Expr<float3> normal, Expr<float3> tangent) noexcept {
        auto bitangent = normalize(cross(normal, tangent));
        auto t = normalize(cross(bitangent, normal));
        return Frame{std::move(t), std::move(bitangent), normal};
    }
    [[nodiscard]] auto local_to_world(Expr<float3> d) const noexcept {
        return d.x * _u + d.y * _v + d.z * _n;
    }
    [[nodiscard]] auto world_to_local(Expr<float3> d) const noexcept {
        using namespace luisa::compute;
        return make_float3(dot(d, _u), dot(d, _v), dot(d, _n));
    }
    [[nodiscard]] Expr<float3> u() const noexcept { return _u; }
    [[nodiscard]] Expr<float3> v() const noexcept { return _v; }
    [[nodiscard]] Expr<float3> n() const noexcept { return _n; }
};

class Interaction {

private:
    Float3 _p;
    Float3 _wo;
    Float3 _ng;
    Float2 _uv;
    Bool _valid;
    Frame _shading;
    Var<InstancedShape> _shape;

public:
    Interaction() noexcept : _valid{false} {}
    Interaction(Var<InstancedShape> shape, Expr<float3> p, Expr<float3> wo, Expr<float3> ng) noexcept
        : _p{p}, _wo{wo}, _ng{ng}, _valid{true},
          _shading{Frame::make(ng)},
          _shape{std::move(shape)} {}
    Interaction(Var<InstancedShape> shape, Expr<float3> p, Expr<float3> wo, Expr<float3> ng, Expr<float2> uv, Expr<float3> ns, Expr<float3> tangent) noexcept
        : _p{p}, _wo{wo}, _ng{ng}, _uv{uv},  _valid{true},
          _shading{Frame::make(ns, tangent)},
          _shape{std::move(shape)} {}
    [[nodiscard]] auto p() const noexcept { return _p; }
    [[nodiscard]] auto ng() const noexcept { return _ng; }
    [[nodiscard]] auto wo() const noexcept { return _wo; }
    [[nodiscard]] auto uv() const noexcept { return _uv; }
    [[nodiscard]] auto valid() const noexcept { return _valid; }
    [[nodiscard]] const auto &shading() const noexcept { return _shading; }
    [[nodiscard]] const auto &shape() const noexcept { return _shape; }
    [[nodiscard]] auto spawn_ray(Expr<float3> wi) const noexcept {
        return luisa::compute::make_ray_robust(_p, _ng, wi);
    }
    [[nodiscard]] auto spawn_ray_to(Expr<float3> p_light, Expr<float> eps = 1e-3f) const noexcept {
        auto l = p_light - _p;
        return luisa::compute::make_ray_robust(_p, _ng, normalize(l), length(l) - eps);
    }
};

}// namespace luisa::render
