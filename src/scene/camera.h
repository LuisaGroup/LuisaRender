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

    class Instance {

    private:
        const Camera *_camera;

    public:
        explicit Instance(const Camera *camera) noexcept : _camera{camera} {}
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto node() const noexcept { return _camera; }

        // generate ray in camera space, should not consider _filter and/or _transform
        [[nodiscard]] virtual Sample generate_ray(
            Sampler::Instance &sampler, Expr<float2> pixel, Expr<float> time) const noexcept = 0;
    };

    struct ShutterPoint {
        float time;
        float weight;
    };

    struct ShutterSample {
        ShutterPoint point;
        uint spp;
    };

private:
    const Film *_film;
    const Filter *_filter;
    const Transform *_transform;
    float2 _shutter_span;
    uint _shutter_samples;
    uint _spp;
    std::filesystem::path _file;
    luisa::vector<ShutterPoint> _shutter_points;

public:
    Camera(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto film() const noexcept { return _film; }
    [[nodiscard]] auto filter() const noexcept { return _filter; }
    [[nodiscard]] auto transform() const noexcept { return _transform; }
    [[nodiscard]] auto shutter_span() const noexcept { return _shutter_span; }
    [[nodiscard]] auto shutter_weight(float time) const noexcept -> float;
    [[nodiscard]] auto shutter_samples() const noexcept -> luisa::vector<ShutterSample>;
    [[nodiscard]] auto spp() const noexcept { return _spp; }
    [[nodiscard]] auto file() const noexcept { return _file; }
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render
