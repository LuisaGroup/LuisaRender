//
// Created by Mike on 2022/1/7.
//

#include <rtx/ray.h>
#include <util/sampling.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class ThinLensCamera final : public Camera {

private:
    float3 _position;
    float3 _look_at;
    float3 _up;
    float _aperture;
    float _focal_length;

public:
    ThinLensCamera(Scene *scene, const SceneNodeDesc *desc) noexcept
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

struct ThinLensCameraData {
    float3 position;
    float3 front;
    float3 up;
    float3 left;
    float2 pixel_offset;
    float2 resolution;
    float focal_plane;
    float lens_radius;
    float projected_pixel_size;
};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::ThinLensCameraData,
             position, front, up, left, pixel_offset, resolution,
             focal_plane, lens_radius, projected_pixel_size){};

namespace luisa::render {

class ThinLensCameraInstance final : public Camera::Instance {

private:
    ThinLensCameraData _host_data{};
    BufferView<ThinLensCameraData> _device_data;

public:
    explicit ThinLensCameraInstance(
        Pipeline &ppl, CommandBuffer &command_buffer,
        const ThinLensCamera *camera) noexcept
        : Camera::Instance{ppl, command_buffer, camera},
          _device_data{ppl.arena_buffer<ThinLensCameraData>(1u)} {
        auto position = camera->position();
        auto v = camera->look_at() - position;
        auto f = camera->focal_length() * 1e-3;
        auto focal_plane = length(v);
        auto sensor_plane = 1. / (1. / f - 1. / focal_plane);
        auto object_to_sensor_ratio = static_cast<float>(
            focal_plane / sensor_plane);
        auto front = normalize(v);
        auto up = camera->up();
        auto left = normalize(cross(up, front));
        up = normalize(cross(front, left));
        auto lens_radius = static_cast<float>(.5 * f / camera->aperture());
        auto resolution = make_float2(camera->film()->resolution());
        auto pixel_offset = .5f * resolution;
        auto projected_pixel_size =
            resolution.x > resolution.y ?
                // landscape mode
                min(static_cast<float>(object_to_sensor_ratio * .036 / resolution.x),
                    static_cast<float>(object_to_sensor_ratio * .024 / resolution.y)) :
                // portrait mode
                min(static_cast<float>(object_to_sensor_ratio * .024 / resolution.x),
                    static_cast<float>(object_to_sensor_ratio * .036 / resolution.y));
        _host_data = {position, front, up, left, pixel_offset, resolution,
                      focal_plane, lens_radius, projected_pixel_size};
        command_buffer << _device_data.copy_from(&_host_data);
    }

private:
    [[nodiscard]] Camera::Sample
    _generate_ray(Sampler::Instance &sampler, Expr<float2> pixel, Expr<float> time) const noexcept override {
        auto data = _device_data.read(0u);
        auto coord_focal = (data.pixel_offset - pixel) * data.projected_pixel_size;
        auto p_focal = coord_focal.x * data.left + coord_focal.y * data.up + data.focal_plane * data.front;
        auto coord_lens = sample_uniform_disk_concentric(sampler.generate_2d()) * data.lens_radius;
        auto p_lens = coord_lens.x * data.left + coord_lens.y * data.up;
        return {.ray = make_ray(p_lens + data.position, normalize(p_focal - p_lens)), .weight = 1.f};
    }
};

luisa::unique_ptr<Camera::Instance> ThinLensCamera::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ThinLensCameraInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ThinLensCamera)
