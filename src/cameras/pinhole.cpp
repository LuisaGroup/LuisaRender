//
// Created by Mike on 2022/1/7.
//

#include <rtx/ray.h>
#include <base/camera.h>
#include <base/film.h>
#include <base/pipeline.h>

namespace luisa::render {

struct PinholeCameraData {
    float3 position;
    float3 front;
    float3 right;
    float3 up;
    float2 resolution;
    float tan_half_fov;
};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::PinholeCameraData,
             position, front, right, up,
             resolution, tan_half_fov){};

namespace luisa::render {

using namespace luisa::compute;

class PinholeCamera;
class PinholeCameraInstance;

class PinholeCamera final : public Camera {

private:
    float3 _position;
    float3 _front;
    float3 _right;
    float3 _up;
    float _fov;

public:
    PinholeCamera(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Camera{scene, desc},
          _position{desc->property_float3("position")},
          _front{desc->property_float3_or_default("front", make_float3(0.0f, 0.0f, -1.0f))},
          _up{desc->property_float3_or_default("up", make_float3(0.0f, 1.0f, 0.0f))},
          _fov{desc->property_float_or_default("fov", 35.0f)} {
        _front = normalize(_front);
        _right = normalize(cross(_front, _up));
        _up = normalize(cross(_right, _front));
        _fov = radians(_fov);
    }

    [[nodiscard]] luisa::unique_ptr<Camera::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto position() const noexcept { return _position; }
    [[nodiscard]] auto front() const noexcept { return _front; }
    [[nodiscard]] auto right() const noexcept { return _right; }
    [[nodiscard]] auto up() const noexcept { return _up; }
    [[nodiscard]] auto fov() const noexcept { return _fov; }
};

class PinholeCameraInstance final : public Camera::Instance {

private:
    PinholeCameraData _host_data;
    BufferView<PinholeCameraData> _device_data;

public:
    explicit PinholeCameraInstance(
        Pipeline &ppl, CommandBuffer &command_buffer,
        const PinholeCamera *camera, float3 position,
        float3 front, float3 up, float3 right, float fov) noexcept
        : Camera::Instance{ppl, command_buffer, camera},
          _host_data{
              .position = position,
              .front = front,
              .right = right,
              .up = up,
              .resolution = make_float2(camera->film()->resolution()),
              .tan_half_fov = tan(fov * 0.5f)},
          _device_data{ppl.arena_buffer<PinholeCameraData>(1u)} {
        command_buffer << _device_data.copy_from(&_host_data);
    }

private:
    [[nodiscard]] Camera::Sample _generate_ray_in_camera_space(
        Sampler::Instance &sampler,
        Expr<float2> pixel, Expr<float> time) const noexcept override;
};

Camera::Sample PinholeCameraInstance::_generate_ray_in_camera_space(
    Sampler::Instance & /* sampler */, Expr<float2> pixel, Expr<float> /* time */) const noexcept {
    auto data = _device_data.read(0u);
    auto p = (pixel * 2.0f - data.resolution) * (data.tan_half_fov / data.resolution.y);
    auto direction = normalize(p.x * data.right - p.y * data.up + data.front);
    return Camera::Sample{make_ray(data.position, direction), pixel, 1.0f};
}

luisa::unique_ptr<Camera::Instance> PinholeCamera::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<PinholeCameraInstance>(
        pipeline, command_buffer, this,
        _position, _front, _up, _right, _fov);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PinholeCamera)
