//
// Created by Mike Smith on 2022/3/26.
//

#include <dsl/rtx/ray.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/pipeline.h>

namespace luisa::render {

struct OrthoCameraData {
    float2 resolution;
    float scale;
};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::OrthoCameraData, resolution, scale){};

namespace luisa::render {

using namespace luisa::compute;

class OrthoCamera;
class OrthoCameraInstance;

class OrthoCamera : public Camera {

private:
    float _zoom;

public:
    OrthoCamera(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Camera{scene, desc}, _zoom{desc->property_float_or_default("zoom", 0.f)} {}
    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool requires_lens_sampling() const noexcept override { return false; }
    [[nodiscard]] auto zoom() const noexcept { return _zoom; }
};

class OrthoCameraInstance : public Camera::Instance {

private:
    BufferView<OrthoCameraData> _device_data;

public:
    explicit OrthoCameraInstance(
        Pipeline &ppl, CommandBuffer &command_buffer,
        const OrthoCamera *camera) noexcept;
    [[nodiscard]] std::pair<Var<Ray>, Float> _generate_ray_in_camera_space(
        Expr<float2> pixel, Expr<float2> /* u_lens */, Expr<float> /* time */) const noexcept override {
        auto data = _device_data->read(0u);
        auto p = (pixel * 2.0f - data.resolution) / data.resolution.y * data.scale;
        auto ray = make_ray(make_float3(p.x, -p.y, 0.f), make_float3(0.f, 0.f, -1.f));
        return std::make_pair(std::move(ray), 1.0f);
    }
};

OrthoCameraInstance::OrthoCameraInstance(
    Pipeline &ppl, CommandBuffer &command_buffer, const OrthoCamera *camera) noexcept
    : Camera::Instance{ppl, command_buffer, camera},
      _device_data{ppl.arena_buffer<OrthoCameraData>(1u)} {
    OrthoCameraData host_data{make_float2(camera->film()->resolution()),
                              std::pow(2.f, camera->zoom())};
    command_buffer << _device_data.copy_from(&host_data)
                   << commit();
}

luisa::unique_ptr<Camera::Instance> OrthoCamera::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<OrthoCameraInstance>(
        pipeline, command_buffer, this);
}

using ClipPlaneOrthoCamera = ClipPlaneCameraWrapper<
    OrthoCamera, OrthoCameraInstance>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ClipPlaneOrthoCamera)
