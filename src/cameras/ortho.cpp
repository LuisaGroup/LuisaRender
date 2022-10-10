//
// Created by Mike Smith on 2022/3/26.
//

#include <rtx/ray.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/pipeline.h>

namespace luisa::render {

struct OrthoCameraData {
    float3 position;
    float3 front;
    float3 left;
    float3 up;
    float2 resolution;
    float scale;
};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::OrthoCameraData,
             position, front, left, up, resolution, scale){};

namespace luisa::render {

using namespace luisa::compute;

class OrthoCamera;
class OrthoCameraInstance;

class OrthoCamera final : public Camera {

private:
    float3 _position;
    float3 _front;
    float3 _left;
    float3 _up;
    float _zoom;

public:
    OrthoCamera(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Camera{scene, desc},
          _position{desc->property_float3("position")},
          _front{desc->property_float3_or_default("front", make_float3(0.0f, 0.0f, -1.0f))},
          _up{desc->property_float3_or_default("up", make_float3(0.0f, 1.0f, 0.0f))},
          _zoom{desc->property_float_or_default("zoom", 0.f)} {
        _front = normalize(_front);
        _left = normalize(cross(_up, _front));
        _up = normalize(cross(_front, _left));
    }

    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto position() const noexcept { return _position; }
    [[nodiscard]] auto front() const noexcept { return _front; }
    [[nodiscard]] auto left() const noexcept { return _left; }
    [[nodiscard]] auto up() const noexcept { return _up; }
    [[nodiscard]] auto zoom() const noexcept { return _zoom; }
};

class OrthoCameraInstance final : public Camera::Instance {

private:
    OrthoCameraData _host_data;
    BufferView<OrthoCameraData> _device_data;

public:
    explicit OrthoCameraInstance(
        Pipeline &ppl, CommandBuffer &command_buffer,
        const OrthoCamera *camera) noexcept;

private:
    [[nodiscard]] Camera::Sample _generate_ray_in_camera_space(
        Sampler::Instance &sampler,
        Expr<float2> pixel, Expr<float> time) const noexcept override;
};

OrthoCameraInstance::OrthoCameraInstance(
    Pipeline &ppl, CommandBuffer &command_buffer, const OrthoCamera *camera) noexcept
    : Camera::Instance{ppl, command_buffer, camera},
      _host_data{.position = camera->position(),
                 .front = camera->front(),
                 .left = camera->left(),
                 .up = camera->up(),
                 .resolution = make_float2(camera->film()->resolution()),
                 .scale = std::pow(2.f, camera->zoom())},
      _device_data{ppl.arena_buffer<OrthoCameraData>(1u)} {
    command_buffer << _device_data.copy_from(&_host_data);
}

Camera::Sample OrthoCameraInstance::_generate_ray_in_camera_space(
    Sampler::Instance & /* sampler */, Expr<float2> pixel, Expr<float> /* time */) const noexcept {
    auto data = _device_data.read(0u);
    auto p = (data.resolution - pixel * 2.0f) / data.resolution.y * data.scale;
    auto u = data.left;
    auto v = data.up;
    auto w = data.front;
    auto origin = p.x * u + p.y * v + data.position;
    return Camera::Sample{make_ray(origin, w), pixel, 1.0f};
}

luisa::unique_ptr<Camera::Instance> OrthoCamera::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<OrthoCameraInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::OrthoCamera)
