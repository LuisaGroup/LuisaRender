//
// Created by Mike on 2022/1/7.
//

#include <dsl/rtx/ray.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/pipeline.h>

namespace luisa::render {

struct PinholeCameraData {
    float2 resolution;
    float tan_half_fov;
};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::PinholeCameraData,
             resolution, tan_half_fov){};

namespace luisa::render {

using namespace luisa::compute;

class PinholeCamera;
class PinholeCameraInstance;

class PinholeCamera : public Camera {

private:
    float _fov;

public:
    PinholeCamera(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Camera{scene, desc},
          _fov{radians(std::clamp(desc->property_float_or_default("fov", 35.0f), 1e-3f, 180.f - 1e-3f))} {}
    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool requires_lens_sampling() const noexcept override { return false; }
    [[nodiscard]] auto fov() const noexcept { return _fov; }
};

class PinholeCameraInstance : public Camera::Instance {

private:
    BufferView<PinholeCameraData> _device_data;

public:
    explicit PinholeCameraInstance(
        Pipeline &ppl, CommandBuffer &command_buffer,
        const PinholeCamera *camera) noexcept
        : Camera::Instance{ppl, command_buffer, camera},
          _device_data{ppl.arena_buffer<PinholeCameraData>(1u)} {
        PinholeCameraData host_data{make_float2(camera->film()->resolution()),
                                    tan(camera->fov() * 0.5f)};
        command_buffer << _device_data.copy_from(&host_data) << commit();
    }
    [[nodiscard]] std::pair<Var<Ray>, Float> _generate_ray_in_camera_space(
        Expr<float2> pixel, Expr<float2> /* u_lens */, Expr<float> /* time */) const noexcept override {
        auto data = _device_data->read(0u);
        auto p = (pixel * 2.0f - data.resolution) * (data.tan_half_fov / data.resolution.y);
        auto direction = normalize(make_float3(p.x, -p.y, -1.f));
        auto ray = make_ray(make_float3(), direction);
        return std::make_pair(std::move(ray), 1.f);
    }
};

luisa::unique_ptr<Camera::Instance> PinholeCamera::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<PinholeCameraInstance>(
        pipeline, command_buffer, this);
}

using ClipPlanePinholeCamera = ClipPlaneCameraWrapper<
    PinholeCamera, PinholeCameraInstance>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ClipPlanePinholeCamera)
