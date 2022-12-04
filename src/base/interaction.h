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

struct RayDifferential {

    Var<Ray> ray;
    Float3 rx_origin;
    Float3 ry_origin;
    Float3 rx_direction;
    Float3 ry_direction;

    void scale_differential(Expr<float> amount) noexcept {
        rx_origin = ray->origin() + (rx_origin - ray->origin()) * amount;
        ry_origin = ray->origin() + (ry_origin - ray->origin()) * amount;
        rx_direction = ray->direction() + (rx_direction - ray->direction()) * amount;
        ry_direction = ray->direction() + (ry_direction - ray->direction()) * amount;
    }

    void scale_differential_uv(Expr<float2> amount_uv) noexcept {
        rx_origin = ray->origin() + (rx_origin - ray->origin()) * amount_uv.x;
        ry_origin = ray->origin() + (ry_origin - ray->origin()) * amount_uv.y;
        rx_direction = ray->direction() + (rx_direction - ray->direction()) * amount_uv.x;
        ry_direction = ray->direction() + (ry_direction - ray->direction()) * amount_uv.y;
    }
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
    [[nodiscard]] Bool same_sided(Expr<float3> wo, Expr<float3> wi) const noexcept;
    [[nodiscard]] Float3 p_robust(Expr<float3> w) const noexcept;
    [[nodiscard]] Var<Ray> spawn_ray(Expr<float3> wi, Expr<float> t_max = std::numeric_limits<float>::max()) const noexcept;
};

}// namespace luisa::render
