//
// Created by Mike on 2022/1/7.
//

#include <dsl/rtx/ray.h>
#include <util/sampling.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class ThinLensCamera : public Camera {

private:
    float _aperture;
    float _focal_length;
    float _focus_distance;

public:
    ThinLensCamera(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Camera{scene, desc},
          _aperture{desc->property_float_or_default("aperture", 2.f)},
          _focal_length{desc->property_float_or_default("focal_length", 35.f)},
          _focus_distance{desc->property_float_or_default(
              "focus_distance", lazy_construct([desc] {
                  auto target = desc->property_float3("look_at");
                  auto position = desc->property_float3("position");
                  return length(target - position);
              }))} {
        _focus_distance = std::max(std::abs(_focus_distance), 1e-4f);
    }
    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool requires_lens_sampling() const noexcept override { return true; }
    [[nodiscard]] auto aperture() const noexcept { return _aperture; }
    [[nodiscard]] auto focal_length() const noexcept { return _focal_length; }
    [[nodiscard]] auto focus_distance() const noexcept { return _focus_distance; }
};

struct ThinLensCameraData {
    float2 pixel_offset;
    float2 resolution;
    float focus_distance;
    float lens_radius;
    float projected_pixel_size;
};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::ThinLensCameraData,
             pixel_offset, resolution, focus_distance,
             lens_radius, projected_pixel_size){};

namespace luisa::render {

class ThinLensCameraInstance : public Camera::Instance {

private:
    BufferView<ThinLensCameraData> _device_data;

public:
    explicit ThinLensCameraInstance(
        Pipeline &ppl, CommandBuffer &command_buffer,
        const ThinLensCamera *camera) noexcept
        : Camera::Instance{ppl, command_buffer, camera},
          _device_data{ppl.arena_buffer<ThinLensCameraData>(1u)} {
        auto v = camera->focus_distance();
        auto f = camera->focal_length() * 1e-3;
        auto u = 1. / (1. / f - 1. / v);// 1 / f = 1 / v + 1 / sensor_plane
        auto object_to_sensor_ratio = static_cast<float>(v / u);
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
        ThinLensCameraData host_data{pixel_offset, resolution, v,
                                     lens_radius, projected_pixel_size};
        command_buffer << _device_data.copy_from(&host_data) << commit();
    }

    [[nodiscard]] std::pair<Var<Ray>, Float> _generate_ray_in_camera_space(Expr<float2> pixel,
                                                                           Expr<float2> u_lens,
                                                                           Expr<float> /* time */) const noexcept override {
        auto data = _device_data->read(0u);
        auto coord_focal = (pixel - data.pixel_offset) * data.projected_pixel_size;
        auto p_focal = make_float3(coord_focal.x, -coord_focal.y, -data.focus_distance);
        auto coord_lens = sample_uniform_disk_concentric(u_lens) * data.lens_radius;
        auto p_lens = make_float3(coord_lens, 0.f);
        auto ray = make_ray(p_lens, normalize(p_focal - p_lens));
        return std::make_pair(std::move(ray), 1.f);
    }
};

luisa::unique_ptr<Camera::Instance> ThinLensCamera::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<ThinLensCameraInstance>(
        pipeline, command_buffer, this);
}

using ClipPlaneThinLensCamera = ClipPlaneCameraWrapper<
    ThinLensCamera, ThinLensCameraInstance>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ClipPlaneThinLensCamera)
