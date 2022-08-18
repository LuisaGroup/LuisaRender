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

class Interaction {

private:
    Var<Shape::Handle> _shape;
    Float3 _pg;
    Float3 _wo;
    Float3 _ng;
    Float2 _uv;
    Float3 _ps;
    Frame _shading;
    UInt _inst_id;
    UInt _prim_id;
    Float _prim_area;

public:
    Interaction() noexcept : _inst_id{~0u}, _prim_id{~0u} {}
    explicit Interaction(Expr<float3> wo) noexcept
        : _wo{wo}, _inst_id{~0u}, _prim_id{~0u} {}
    Interaction(Expr<float3> wo, Expr<float2> uv) noexcept
        : _wo{wo}, _uv{uv}, _inst_id{~0u}, _prim_id{~0u} {}
    Interaction(Var<Shape::Handle> shape, Expr<uint> inst_id, Expr<uint> prim_id, Expr<float> prim_area,
                Expr<float3> p, Expr<float3> wo, Expr<float3> ng) noexcept
        : _shape{std::move(shape)}, _pg{p}, _wo{wo}, _ng{ng}, _shading{Frame::make(_ng)},
          _inst_id{~0u}, _prim_id{prim_id}, _prim_area{prim_area} {}
    Interaction(Var<Shape::Handle> shape, Expr<uint> inst_id, Expr<uint> prim_id, Expr<float> prim_area,
                Expr<float3> pg, Expr<float3> wo, Expr<float3> ng, Expr<float2> uv,
                Expr<float3> ps, Expr<float3> ns, Expr<float3> tangent) noexcept
        : _shape{std::move(shape)}, _pg{pg}, _wo{wo}, _ng{ng}, _uv{uv}, _ps{ps},
          _shading{Frame::make(ite(_shape->two_sided() & (dot(ns, wo) < 0.0f), -ns, ns), tangent)},
          _inst_id{inst_id}, _prim_id{prim_id}, _prim_area{prim_area} {}
    Interaction(Var<Shape::Handle> shape, Expr<uint> inst_id, Expr<uint> prim_id,
                Expr<float3> wo, const ShadingAttribute &attrib) noexcept
        : Interaction{std::move(shape), inst_id, prim_id, attrib.area, attrib.pg,
                      wo, attrib.ng, attrib.uv, attrib.ps, attrib.ns, attrib.tangent} {}
    [[nodiscard]] auto p() const noexcept { return _pg; }
    [[nodiscard]] auto ng() const noexcept { return _ng; }
    [[nodiscard]] auto wo() const noexcept { return _wo; }
    [[nodiscard]] auto uv() const noexcept { return _uv; }
    [[nodiscard]] auto instance_id() const noexcept { return _inst_id; }
    [[nodiscard]] auto triangle_id() const noexcept { return _prim_id; }
    [[nodiscard]] auto triangle_area() const noexcept { return _prim_area; }
    [[nodiscard]] auto valid() const noexcept { return _inst_id != ~0u; }
    [[nodiscard]] auto p_shading() const noexcept { return _ps; }
    [[nodiscard]] const auto &shading() const noexcept { return _shading; }
    void set_shading(Frame frame) noexcept { _shading = std::move(frame); }
    [[nodiscard]] const auto &shape() const noexcept { return _shape; }
    [[nodiscard]] auto wo_local() const noexcept { return _shading.world_to_local(_wo); }
    [[nodiscard]] auto p_robust(Expr<float3> w) const noexcept {
        return ite(dot(_ng, w) < 0.f,
                   offset_ray_origin(_pg, -_ng),
                   offset_ray_origin(_pg, _ng));
    }
    [[nodiscard]] auto spawn_ray(Expr<float3> wi, Expr<float> t_max = std::numeric_limits<float>::max()) const noexcept {
        return make_ray(p_robust(wi), wi, 0.f, t_max);
    }
};

}// namespace luisa::render
