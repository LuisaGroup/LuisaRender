//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <rtx/mesh.h>
#include <base/scene_node.h>

namespace luisa::render {

class Light;
class Material;
class Transform;

using compute::Triangle;

struct Vertex {
    float p[3];
    float n[3];
    float t[2];
};

class Shape : public SceneNode {

private:
    const Material *_material;
    const Light *_light;
    const Transform *_transform;

public:
    Shape(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto material() const noexcept { return _material; }
    [[nodiscard]] auto light() const noexcept { return _light; }
    [[nodiscard]] auto transform() const noexcept { return _transform; }
    [[nodiscard]] virtual bool is_static() const noexcept = 0;
    [[nodiscard]] virtual bool is_rigid() const noexcept = 0;
    [[nodiscard]] virtual bool is_mesh() const noexcept = 0;
    [[nodiscard]] virtual std::span<const Vertex> vertices() const noexcept = 0;
    [[nodiscard]] virtual std::span<const Triangle> triangles() const noexcept = 0;
    [[nodiscard]] virtual size_t child_count() const noexcept = 0;
    [[nodiscard]] virtual const Shape *child(size_t index) const noexcept = 0;
};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::Vertex, p, n, t) {
    [[nodiscard]] auto position() const noexcept {
        return luisa::compute::make_float3(p[0], p[1], p[2]);
    }
    [[nodiscard]] auto normal() const noexcept {
        return luisa::compute::make_float3(n[0], n[1], n[2]);
    }
    [[nodiscard]] auto uv() const noexcept {
        return luisa::compute::make_float2(t[0], t[1]);
    }
    void set_position(luisa::compute::Expr<luisa::float3> v) noexcept {
        p[0] = v.x;
        p[1] = v.y;
        p[2] = v.z;
    }
    void set_normal(luisa::compute::Expr<luisa::float3> v) noexcept {
        n[0] = v.x;
        n[1] = v.y;
        n[2] = v.z;
    }
    void set_uv(luisa::compute::Expr<luisa::float2> v) noexcept {
        t[0] = v.x;
        t[1] = v.y;
    }
};
