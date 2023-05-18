//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <dsl/syntax.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <base/film.h>
#include <base/filter.h>
#include <base/texture.h>
#include <base/interaction.h>

namespace luisa::render {

using compute::Expr;
using compute::Float3;
using compute::Float4x4;
using compute::Image;
using compute::Ray;
using compute::Var;

class Sampler;
class Transform;

class Camera : public SceneNode {

public:
    struct Sample {
        Var<Ray> ray;
        Float2 pixel;
        Float weight;
    };

    struct SampleDifferential {
        RayDifferential ray_differential;
        Float2 pixel;
        Float weight;
    };

    class Instance {

    private:
        const Pipeline *_pipeline;
        const Camera *_camera;
        luisa::unique_ptr<Film::Instance> _film;
        const Filter::Instance *_filter;

    private:
        // generate ray in camera space, should not consider _filter or _transform
        [[nodiscard]] virtual std::pair<Var<Ray>, Float /* weight */>
        _generate_ray_in_camera_space(Expr<float2> pixel,
                                      Expr<float2> u_lens,
                                      Expr<float> time) const noexcept = 0;

    public:
        Instance(Pipeline &pipeline,
                 CommandBuffer &command_buffer,
                 const Camera *camera) noexcept;
        Instance(const Instance &) noexcept = delete;
        Instance(Instance &&another) noexcept = default;
        Instance &operator=(const Instance &) noexcept = delete;
        Instance &operator=(Instance &&) noexcept = delete;
        virtual ~Instance() noexcept = default;
        template<typename T = Camera>
            requires std::is_base_of_v<Camera, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_camera); }
        [[nodiscard]] auto &pipeline() const noexcept { return *_pipeline; }
        [[nodiscard]] auto film() noexcept { return _film.get(); }
        [[nodiscard]] auto film() const noexcept { return _film.get(); }
        [[nodiscard]] auto filter() noexcept { return _filter; }
        [[nodiscard]] auto filter() const noexcept { return _filter; }
        [[nodiscard]] Sample generate_ray(Expr<uint2> pixel_coord, Expr<float> time,
                                          Expr<float2> u_filter, Expr<float2> u_lens) const noexcept;
        [[nodiscard]] SampleDifferential generate_ray_differential(Expr<uint2> pixel_coord, Expr<float> time,
                                                                   Expr<float2> u_filter, Expr<float2> u_lens) const noexcept;
        [[nodiscard]] Float4x4 camera_to_world() const noexcept;
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
    [[nodiscard]] virtual bool requires_lens_sampling() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

template<typename Base,
         typename BaseInstance = typename Base::Instance>
class ClipPlaneCameraWrapper : public Base {

private:
    float2 _clip_plane;

public:
    ClipPlaneCameraWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Base{scene, desc},
          _clip_plane{desc->property_float2_or_default(
              "clip", lazy_construct([desc] {
                  return desc->property_float2_or_default(
                      "clip_plane", lazy_construct([desc] {
                          auto near_plane = desc->property_float_or_default(
                              "clip", lazy_construct([desc] {
                                  return desc->property_float_or_default("clip_plane", 0.f);
                              }));
                          return make_float2(near_plane, 1e10f);
                      }));
              }))} {
        _clip_plane = clamp(_clip_plane, 0.f, 1e10f);
        if (_clip_plane.x > _clip_plane.y) { std::swap(_clip_plane.x, _clip_plane.y); }
    }

public:
    class Instance : public BaseInstance {

    public:
        explicit Instance(BaseInstance base) noexcept
            : BaseInstance{std::move(base)} {}
        [[nodiscard]] std::pair<Var<Ray>, Float>
        _generate_ray_in_camera_space(Expr<float2> pixel,
                                      Expr<float2> u_lens,
                                      Expr<float> time) const noexcept override {
            auto [ray, weight] = BaseInstance::_generate_ray_in_camera_space(pixel, u_lens, time);
            auto t = this->template node<ClipPlaneCameraWrapper>()->clip_plane() /
                     dot(ray->direction(), make_float3(0.f, 0.f, -1.f));
            ray->set_t_min(t.x);
            ray->set_t_max(t.y);
            return std::make_pair(std::move(ray), std::move(weight));
        }
    };
    [[nodiscard]] auto clip_plane() const noexcept { return _clip_plane; }
    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto base = Base::build(pipeline, command_buffer);
        return luisa::make_unique<ClipPlaneCameraWrapper::Instance>(
            std::move(dynamic_cast<BaseInstance &>(*base)));
    }
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(::luisa::render::Camera::Instance)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(::luisa::render::Camera::Sample)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(::luisa::render::Camera::SampleDifferential)
