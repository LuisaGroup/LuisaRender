//
// Created by Mike on 2022/1/7.
//

#include <rtx/ray.h>
#include <util/sampling.h>
#include <base/camera.h>
#include <base/film.h>

namespace luisa::render {

using namespace luisa::compute;

class ThinlensCamera final : public Camera {

private:
    float3 _position;
    float3 _look_at;
    float3 _up;
    float _aperture;
    float _focal_length;

public:
    ThinlensCamera(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Camera{scene, desc},
          _position{desc->property_float3("position")},
          _look_at{desc->property_float3("look_at")},
          _up{desc->property_float3_or_default("up", make_float3(0.0f, 1.0f, 0.0f))},
          _aperture{desc->property_float_or_default("aperture", 2.f)},
          _focal_length{desc->property_float_or_default("focal_length", 35.f)} {}

    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto position() const noexcept { return _position; }
    [[nodiscard]] auto look_at() const noexcept { return _look_at; }
    [[nodiscard]] auto up() const noexcept { return _up; }
    [[nodiscard]] auto aperture() const noexcept { return _aperture; }
    [[nodiscard]] auto focal_length() const noexcept { return _focal_length; }
};

class ThinlensCameraInstance final : public Camera::Instance {

private:
    float3 _position;
    float3 _front;
    float3 _up;
    float3 _left;
    float2 _pixel_offset;
    float _focal_plane{};
    float _lens_radius{};
    float _projected_pixel_size{};

public:
    explicit ThinlensCameraInstance(
        Pipeline &ppl, CommandBuffer &command_buffer,
        const ThinlensCamera *camera) noexcept
        : Camera::Instance{ppl, command_buffer, camera} {
        _position = camera->position();
        auto v = camera->look_at() - _position;
        auto f = camera->focal_length() * 1e-3;
        _focal_plane = length(v);
        auto sensor_plane = 1. / (1. / f - 1. / _focal_plane);
        auto object_to_sensor_ratio = static_cast<float>(
            _focal_plane / sensor_plane);
        _front = normalize(v);
        _up = camera->up();
        _left = normalize(cross(_up, _front));
        _up = normalize(cross(_front, _left));
        _lens_radius = static_cast<float>(.5 * f / camera->aperture());
        auto resolution = make_float2(camera->film()->resolution());
        _pixel_offset = .5f * resolution;
        if (resolution.x > resolution.y) {// landscape mode
            _projected_pixel_size = min(
                static_cast<float>(object_to_sensor_ratio * .036 / resolution.x),
                static_cast<float>(object_to_sensor_ratio * .024 / resolution.y));
        } else {// portrait mode
            _projected_pixel_size = min(
                static_cast<float>(object_to_sensor_ratio * .024 / resolution.x),
                static_cast<float>(object_to_sensor_ratio * .036 / resolution.y));
        }
    }

private:
    [[nodiscard]] Camera::Sample _generate_ray(
        Sampler::Instance &sampler,
        Expr<float2> pixel, Expr<float> time) const noexcept override {
        auto coord_focal = (_pixel_offset - pixel) * _projected_pixel_size;
        auto p_focal = coord_focal.x * _left + coord_focal.y * _up + _focal_plane * _front;
        auto coord_lens = sample_uniform_disk_concentric(sampler.generate_2d()) * _lens_radius;
        auto p_lens = coord_lens.x * _left + coord_lens.y * _up;
        return {.ray = make_ray(p_lens + _position, normalize(p_focal - p_lens)), .weight = 1.f};
    }
};

luisa::unique_ptr<Camera::Instance> ThinlensCamera::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ThinlensCameraInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ThinlensCamera)
