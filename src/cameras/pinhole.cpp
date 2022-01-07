//
// Created by Mike on 2022/1/7.
//

#include <rtx/ray.h>
#include <scene/camera.h>
#include <scene/film.h>

namespace luisa::render {

class PinholeCamera;
class PinholeCameraInstance;

class PinholeCameraInstance final : public Camera::Instance {

private:
    const PinholeCamera *_camera;

public:
    explicit PinholeCameraInstance(const PinholeCamera *camera) noexcept
        : _camera{camera} {}
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
        Stream &stream, Pipeline &pipeline, float initial_time) const noexcept override {
        return luisa::make_unique<PinholeCameraInstance>(this);
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "pinhole"; }
    [[nodiscard]] auto position() const noexcept { return _position; }
    [[nodiscard]] auto front() const noexcept { return _front; }
    [[nodiscard]] auto right() const noexcept { return _right; }
    [[nodiscard]] auto up() const noexcept { return _up; }
    [[nodiscard]] auto fov() const noexcept { return _fov; }
};

Camera::Sample PinholeCameraInstance::generate_ray(
    Sampler::Instance &sampler, Expr<float2> pixel, Expr<float> time [[maybe_unused]]) const noexcept {

    auto resolution = _camera->film()->resolution();
    auto u = sampler.generate_2d();

    // TODO...
    return Camera::Sample{};
}

}// namespace luisa::render

LUISA_SCENE_NODE_MAKE_PLUGIN_API(luisa::render::PinholeCamera)
