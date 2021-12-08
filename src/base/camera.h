//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <dsl/syntax.h>
#include <rtx/ray.h>
#include <base/scene.h>

namespace luisa::render {

using compute::Ray;
using compute::Var;
using compute::Expr;
using compute::Float3;

class Sampler;
class Film;
class Filter;
class Transform;

class Camera : public Scene::Node {

public:
    struct Sample {
        Var<Ray> ray;
        Float3 weight;
    };

private:
    Film *_film{nullptr};
    Filter *_filter{nullptr};
    Transform *_transform{nullptr};

private:
    virtual void _on_set_film() noexcept {}
    virtual void _on_set_filter() noexcept {}
    virtual void _on_set_transform() noexcept {}

public:
    Camera() noexcept : Scene::Node{Scene::Node::Tag::CAMERA} {}
    Camera &set_film(Film *film) noexcept;
    Camera &set_filter(Filter *filter) noexcept;
    Camera &set_transform(Transform *transform) noexcept;
    [[nodiscard]] auto film() const noexcept { return _film; }
    [[nodiscard]] auto filter() const noexcept { return _filter; }
    [[nodiscard]] auto transform() const noexcept { return _transform; }
    [[nodiscard]] virtual Sample generate_ray(Sampler &sampler, Expr<float2> pixel, float time) const noexcept = 0;
};

}
