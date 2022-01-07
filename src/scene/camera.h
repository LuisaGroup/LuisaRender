//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <dsl/syntax.h>
#include <rtx/ray.h>
#include <scene/scene_node.h>
#include <scene/sampler.h>

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

    class Instance : public SceneNode::Instance {

    private:
        const Camera *_camera;

    public:
        explicit Instance(const Camera *camera) noexcept : _camera{camera} {}
        [[nodiscard]] auto camera() const noexcept { return _camera; }

        // generate ray in camera space, should not consider _filter and/or _transform
        [[nodiscard]] virtual Sample generate_ray(
            Sampler::Instance &sampler, Expr<float2> pixel, Expr<float> time) const noexcept = 0;
    };

private:
    const Film *_film;
    const Filter *_filter;
    const Transform *_transform;
    float2 _time_span;

public:
    Camera(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto film() const noexcept { return _film; }
    [[nodiscard]] auto filter() const noexcept { return _filter; }
    [[nodiscard]] auto transform() const noexcept { return _transform; }
    [[nodiscard]] auto time_span() const noexcept { return _time_span; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Stream &stream, Pipeline &pipeline) const noexcept = 0;
};

}// namespace luisa::render
