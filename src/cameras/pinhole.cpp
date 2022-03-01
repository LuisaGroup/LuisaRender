//
// Created by Mike on 2022/1/7.
//

#include <rtx/ray.h>
#include <base/camera.h>
#include <base/film.h>

namespace luisa::render {

using namespace luisa::compute;

class PinholeCamera;
class PinholeCameraInstance;

class PinholeCameraInstance final : public Camera::Instance {

private:
    float3 _position;
    float3 _front;
    float3 _up;
    float3 _right;
    float _fov;

public:
    explicit PinholeCameraInstance(
        const Pipeline &ppl, const PinholeCamera *camera, float3 position,
        float3 front, float3 up, float3 right, float fov) noexcept;

    [[nodiscard]] Camera::Sample generate_ray(
        Sampler::Instance &sampler,
        Expr<float2> pixel, Expr<float> time) const noexcept override;
};

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
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<PinholeCameraInstance>(
            pipeline, this, _position, _front, _up, _right, _fov);
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto position() const noexcept { return _position; }
    [[nodiscard]] auto front() const noexcept { return _front; }
    [[nodiscard]] auto right() const noexcept { return _right; }
    [[nodiscard]] auto up() const noexcept { return _up; }
    [[nodiscard]] auto fov() const noexcept { return _fov; }
};

PinholeCameraInstance::PinholeCameraInstance(
    const Pipeline &ppl, const PinholeCamera *camera, float3 position,
    float3 front, float3 up, float3 right, float fov) noexcept
    : Camera::Instance{ppl, camera}, _position{position},
      _front{front}, _up{up}, _right{right}, _fov{fov} {}

Camera::Sample PinholeCameraInstance::generate_ray(
    Sampler::Instance & /* sampler */, Expr<float2> pixel, Expr<float> /* time */) const noexcept {
    auto resolution = make_float2(node()->film()->resolution());
    auto p = (pixel * 2.0f - resolution) * (std::tan(_fov * 0.5f) / resolution.y);
    auto direction = normalize(p.x * _right - p.y * _up + _front);
    return Camera::Sample{make_ray(_position, direction), 1.0f};
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PinholeCamera)
