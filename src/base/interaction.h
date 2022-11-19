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

struct ShadingAttribute {
    Float3 pg;
    Float3 ng;
    Float3 ps;
    Float3 ns;
    Float3 tangent;
    Float2 uv;
    Float area;
};

class RayDifferential {

private:
    Var<Ray> _ray;
    Float3 _rx_origin;
    Float3 _ry_origin;
    Float3 _rx_direction;
    Float3 _ry_direction;

public:
};

class Interaction {

private:
    Var<Shape::Handle> _shape;
    Float3 _pg;
    Float3 _ng;
    Float2 _uv;
    Float3 _ps;
    Frame _shading;
    UInt _inst_id;
    UInt _prim_id;
    Float _prim_area;
    Bool _back_facing;

public:
    Interaction() noexcept : _inst_id{~0u}, _prim_id{~0u} {}
    explicit Interaction(Expr<float2> uv) noexcept : _uv{uv}, _inst_id{~0u}, _prim_id{~0u} {}

    Interaction(Var<Shape::Handle> shape, Expr<uint> inst_id, Expr<uint> prim_id, Expr<float> prim_area,
                Expr<float3> p, Expr<float3> ng, Expr<bool> back_facing) noexcept
        : _shape{std::move(shape)}, _pg{p}, _ng{ng}, _shading{Frame::make(_ng)}, _ps{p},
          _inst_id{~0u}, _prim_id{prim_id}, _prim_area{prim_area}, _back_facing{back_facing} {}

    Interaction(Var<Shape::Handle> shape, Expr<uint> inst_id, Expr<uint> prim_id,
                Expr<float> prim_area, Expr<float3> pg, Expr<float3> ng, Expr<float2> uv,
                Expr<float3> ps, Expr<float3> ns, Expr<float3> tangent, Expr<bool> back_facing) noexcept
        : _shape{std::move(shape)}, _pg{pg}, _ng{ng}, _uv{uv}, _ps{ps}, _shading{Frame::make(ns, tangent)},
          _inst_id{inst_id}, _prim_id{prim_id}, _prim_area{prim_area}, _back_facing{back_facing} {}

    Interaction(Var<Shape::Handle> shape, Expr<uint> inst_id, Expr<uint> prim_id,
                const ShadingAttribute &attrib, Expr<bool> back_facing) noexcept
        : Interaction{std::move(shape), inst_id, prim_id, attrib.area, attrib.pg, attrib.ng,
                      attrib.uv, attrib.ps, attrib.ns, attrib.tangent, back_facing} {}

    [[nodiscard]] auto p() const noexcept { return _pg; }
    [[nodiscard]] auto p_shading() const noexcept { return _ps; }
    [[nodiscard]] auto ng() const noexcept { return _ng; }
    [[nodiscard]] auto uv() const noexcept { return _uv; }
    [[nodiscard]] auto instance_id() const noexcept { return _inst_id; }
    [[nodiscard]] auto triangle_id() const noexcept { return _prim_id; }
    [[nodiscard]] auto triangle_area() const noexcept { return _prim_area; }
    [[nodiscard]] auto valid() const noexcept { return _inst_id != ~0u; }
    [[nodiscard]] const auto &shading() const noexcept { return _shading; }
    void set_shading(Frame frame) noexcept { _shading = std::move(frame); }
    [[nodiscard]] const auto &shape() const noexcept { return _shape; }
    [[nodiscard]] auto back_facing() const noexcept { return _back_facing; }
    [[nodiscard]] auto p_robust(Expr<float3> w) const noexcept {
        auto offset_factor = clamp(_shape->intersection_offset_factor() * 15.f + 1.f, 1.f, 16.f);
        //        return ite(dot(_ng, w) < 0.f,
        //                   offset_ray_origin(_pg, -_ng),
        //                   offset_ray_origin(_ps, _ng));
        // switched to the following method to match Mitsuba 3
        // Float mag = (1.f + dr::max(dr::abs(p))) * math::RayEpsilon<Float>;
        //        mag = dr::detach(dr::mulsign(mag, dr::dot(n, d)));
        //        return dr::fmadd(mag, dr::detach(n), p);
        auto max_comp = [](auto v) noexcept { return max(max(v.x, v.y), v.z); };
        constexpr auto ray_eps = 0x1p-24f * 1.5e3f;
        auto front = dot(_ng, w) > 0.f;
        auto mag = (1.f + max_comp(abs(_pg))) * ray_eps * offset_factor * ite(front, 1.f, -1.f);
        return fma(mag, _ng, ite(front, _ps, _pg));
    }
    [[nodiscard]] auto spawn_ray(Expr<float3> wi, Expr<float> t_max = std::numeric_limits<float>::max()) const noexcept {
        return make_ray(p_robust(wi), wi, 0.f, t_max);
    }
};

}// namespace luisa::render
