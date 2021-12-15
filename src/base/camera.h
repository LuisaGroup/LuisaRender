//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <dsl/syntax.h>
#include <rtx/ray.h>
#include <base/scene_node.h>

namespace luisa::render {

using compute::Expr;
using compute::Float3;
using compute::Ray;
using compute::Var;

class Sampler;
class Film;
class Filter;
class Transform;

class Camera : public SceneNode {

public:
    struct Sample {
        Var<Ray> ray;
        Float3 weight;
    };

private:
    const Film *_film;
    const Filter *_filter;
    const Transform *_transform;

public:
    Camera(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto film() const noexcept { return _film; }
    [[nodiscard]] auto filter() const noexcept { return _filter; }
    [[nodiscard]] auto transform() const noexcept { return _transform; }

    // generate a ray in the **camera** space; the renderer is responsible for the camera-to-world transform
    // camera space: right-handed, x - right, y - up, z - out
    [[nodiscard]] virtual Sample generate_ray(Sampler &sampler, Expr<uint2> pixel, Expr<float> time) const noexcept = 0;
};

}// namespace luisa::render
