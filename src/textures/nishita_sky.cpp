//
// Created by Mike Smith on 2022/10/13.
//

#include <base/texture.h>
#include <base/pipeline.h>
#include <textures/sky_precompute.h>

namespace luisa::render {

class NishitaSky final : public Texture {

public:
    static constexpr auto resolution = make_uint2(2048u);
    static constexpr auto height_per_thread = 16u;

private:
    float _sun_angle;
    float _sun_elevation;
    float _altitude;
    float _air_density;
    float _dust_density;
    float _ozone_density;
    float _sun_intensity;
    float _scale;
    LoadedImage _image;
    std::atomic_uint _image_counter{0u};
    luisa::optional<NishitaSkyPrecomputedSun> _sun{};

public:
    NishitaSky(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _sun_angle{radians(std::clamp(desc->property_float_or_default("sun_angle", .545f), 1e-3f, 360.f))},
          _sun_elevation{radians(std::clamp(desc->property_float_or_default("sun_elevation", 15.f), 0.f, 90.f))},
          _altitude{std::clamp(desc->property_float_or_default("altitude", 1.f), 1.f, 59.999e3f)},
          _air_density{std::clamp(desc->property_float_or_default("air_density", 1.f), 0.f, 10.f)},
          _dust_density{std::clamp(desc->property_float_or_default("dust_density", 1.f), 0.f, 10.f)},
          _ozone_density{std::clamp(desc->property_float_or_default("ozone_density", 1.f), 0.f, 10.f)},
          _sun_intensity{std::max(desc->property_float_or_default("sun_intensity", 1.f), 0.f)},
          _scale{std::max(desc->property_float_or_default("scale", 1.f), 0.f)} {

        NishitaSkyData data{.sun_elevation = _sun_elevation,
                            .sun_angle = _sun_angle,
                            .altitude = _altitude,
                            .air_density = _air_density,
                            .dust_density = _dust_density,
                            .ozone_density = _ozone_density};
        _image = LoadedImage::create(resolution, PixelStorage::FLOAT4);
        ThreadPool::global().parallel(
            resolution.y / height_per_thread, [data, this](uint32_t y) noexcept {
                SKY_nishita_skymodel_precompute_texture(
                    data, static_cast<float4 *>(_image.pixels()),
                    resolution, make_uint2(y * height_per_thread, (y + 1u) * height_per_thread));
                _image_counter.fetch_add(1u);
            });
        if (desc->property_bool_or_default("sun_disc", true)) {
            _sun.emplace(SKY_nishita_skymodel_precompute_sun(data));
        }
    }
    [[nodiscard]] auto sun_angle() const noexcept { return _sun_angle; }
    [[nodiscard]] auto sun_elevation() const noexcept { return _sun_elevation; }
    [[nodiscard]] auto altitude() const noexcept { return _altitude; }
    [[nodiscard]] auto air_density() const noexcept { return _air_density; }
    [[nodiscard]] auto dust_density() const noexcept { return _dust_density; }
    [[nodiscard]] auto ozone_density() const noexcept { return _ozone_density; }
    [[nodiscard]] auto sun_intensity() const noexcept { return _sun_intensity; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto &image() const noexcept {
        while (_image_counter.load() < resolution.y / height_per_thread) {
            LUISA_WARNING_WITH_LOCATION(
                "NishitaSky texture is still being precomputed.");
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        return _image;
    }
    [[nodiscard]] auto sun() const noexcept { return _sun; }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.f; }
    [[nodiscard]] uint channels() const noexcept override { return 3u; }
    [[nodiscard]] bool is_constant() const noexcept override { return false; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class NishitaSkyInstance final : public Texture::Instance {

private:
    uint _texture_id{};
    mutable luisa::unique_ptr<Callable<float3(float2)>> _impl;

public:
    NishitaSkyInstance(Pipeline &pipeline, const NishitaSky *node,
                       CommandBuffer &command_buffer) noexcept
        : Texture::Instance{pipeline, node} {
        auto texture = pipeline.create<Image<float>>(
            PixelStorage::FLOAT4, NishitaSky::resolution);
        _texture_id = pipeline.register_bindless(
            *texture, TextureSampler::linear_point_mirror());
        auto &&image = node->image();
        command_buffer << texture->copy_from(image.pixels());
    }
    [[nodiscard]] auto _eval_impl() const noexcept {
        return [this](Float2 uv) noexcept {
            auto geographical_to_direction = [](Float2 latlon) noexcept {
                auto lat = latlon.x;
                auto lon = latlon.y;
                return make_float3(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat));
            };
            auto uv_to_geographical = [](Float2 uv) noexcept {
                auto phi = 2.f * pi * (1.f - uv.x);
                auto theta = pi * uv.y;
                return make_float2(pi_over_two - theta, phi);
            };
            auto sky = node<NishitaSky>();
            auto scale = sky->scale();
            auto latlon = uv_to_geographical(uv);
            auto w = geographical_to_direction(latlon);
            auto v = def(make_float3());
            $if(w.z >= 0.f) {// above the horizon
                if (auto sun = sky->sun()) {
                    auto sun_elev = sky->sun_elevation();
                    auto sun_direction = make_float3(
                        std::cos(sun_elev), 0.f, std::sin(sun_elev));
                    auto half_angle = .5f * sky->sun_angle();
                    auto cos_angle = dot(w, sun_direction);
                    $if(cos_angle > std::cos(half_angle)) {// inside sun disc
                        $if(sun_elev + half_angle > 0.f) {
                            $if(sun_elev - half_angle > 0.f) {
                                auto y = ((latlon.x - sun_elev) / sky->sun_angle()) + .5f;
                                v = lerp(sun->bottom, sun->top, y) * sky->sun_intensity();
                            }
                            $else {
                                auto y = latlon.x / (sun_elev + half_angle);
                                v = lerp(sun->bottom, sun->top, y) * sky->sun_intensity();
                            };
                        };
                        // limb darkening, coefficient = 0.6
                        auto limb_darkening = (1.f - .6f * (1.f - sqrt(1.f - sqr(acos(cos_angle) / half_angle))));
                        v *= limb_darkening;
                    }
                    $else {// outside sun disc, only sky
                        auto x = latlon.y * inv_pi;
                        /* more pixels toward horizon compensation */
                        auto y = sqrt(max(latlon.x * two_over_pi, 0.f));
                        v = pipeline().tex2d(_texture_id).sample(make_float2(x, y)).xyz();
                    };
                } else {// sky only
                    auto x = latlon.y * inv_pi;
                    /* more pixels toward horizon compensation */
                    auto y = sqrt(max(latlon.x * two_over_pi, 0.f));
                    v = pipeline().tex2d(_texture_id).sample(make_float2(x, y)).xyz();
                }
            }
            $elif(w.z >= -.4f) {// below the horizon, but not too far, fade the black ground
                auto fade = 1.f + w.z * 2.5f;
                fade = sqr(fade) * fade;
                /* interpolation */
                auto x = latlon.y * inv_pi;
                v = fade * pipeline().tex2d(_texture_id).sample(make_float2(x, 0.f)).xyz();
            };
            return v * scale;
        };
    }
    [[nodiscard]] Float4 evaluate(const Interaction &it,
                                  const SampledWavelengths &swl,
                                  Expr<float> time) const noexcept override {
        if (_impl == nullptr) {
            _impl = luisa::make_unique<Callable<float3(float2)>>(_eval_impl());
        }
        return make_float4((*_impl)(it.uv()), 1.f);
    }
};

luisa::unique_ptr<Texture::Instance> NishitaSky::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<NishitaSkyInstance>(
        pipeline, this, command_buffer);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NishitaSky)
