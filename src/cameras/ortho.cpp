//
// Created by Mike Smith on 2022/3/26.
//

#include <rtx/ray.h>
#include <base/camera.h>
#include <base/film.h>

namespace luisa::render {

using namespace luisa::compute;

class OrthoCamera;
class OrthoCameraInstance;

class OrthoCameraInstance final : public Camera::Instance {

public:
    explicit OrthoCameraInstance(
        Pipeline &ppl, CommandBuffer &command_buffer,
        const OrthoCamera *camera) noexcept;

private:
    [[nodiscard]] Camera::Sample _generate_ray(
        Sampler::Instance &sampler,
        Expr<float2> pixel, Expr<float> time) const noexcept override;
};

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
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<OrthoCameraInstance>(
            pipeline, command_buffer, this);
    }

    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto position() const noexcept { return _position; }
    [[nodiscard]] auto front() const noexcept { return _front; }
    [[nodiscard]] auto left() const noexcept { return _left; }
    [[nodiscard]] auto up() const noexcept { return _up; }
    [[nodiscard]] auto zoom() const noexcept { return _zoom; }
};

OrthoCameraInstance::OrthoCameraInstance(
    Pipeline &ppl, CommandBuffer &command_buffer, const OrthoCamera *camera) noexcept
    : Camera::Instance{ppl, command_buffer, camera} {}

Camera::Sample OrthoCameraInstance::_generate_ray(
    Sampler::Instance & /* sampler */, Expr<float2> pixel, Expr<float> /* time */) const noexcept {
    auto resolution = make_float2(node()->film()->resolution());
    auto position = node<OrthoCamera>()->position();
    auto scale = std::pow(2.f, -node<OrthoCamera>()->zoom());
    auto p = (resolution - pixel * 2.0f) / resolution.y * scale;
    auto u = node<OrthoCamera>()->left();
    auto v = node<OrthoCamera>()->up();
    auto w = node<OrthoCamera>()->front();
    auto origin = p.x * u + p.y * v + position;
    return Camera::Sample{make_ray(origin, w), 1.0f};
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::OrthoCamera)
