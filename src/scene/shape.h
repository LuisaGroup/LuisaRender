//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <rtx/mesh.h>
#include <scene/scene_node.h>

namespace luisa::render {

struct alignas(16) Instance {

    // vertex buffer information
    uint position_buffer_id;
    uint normal_buffer_id;
    uint tangent_buffer_id;
    uint uv_buffer_id;// 16B

    // index buffer information
    uint triangle_buffer_id;
    uint triangle_count;// 24B

    // transforming & sampling
    uint transform_buffer_id;
    uint area_cdf_buffer_id;// 32B

    // appearance & illumination
    uint material_tag;
    uint material_buffer_id;
    uint light_tag;
    uint light_buffer_id;// 48B
};

static_assert(sizeof(Instance) == 48);

class Light;
class Material;
class Transform;

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
    [[nodiscard]] virtual size_t child_count() const noexcept = 0;
    [[nodiscard]] virtual const Shape *child(size_t index) const noexcept = 0;
};

}// namespace luisa::render

LUISA_STRUCT(
    luisa::render::Instance,

    position_buffer_id,
    normal_buffer_id,
    tangent_buffer_id,
    uv_buffer_id,

    triangle_buffer_id,
    triangle_count,
    transform_buffer_id,
    area_cdf_buffer_id,

    material_tag,
    material_buffer_id,
    light_tag,
    light_buffer_id){};