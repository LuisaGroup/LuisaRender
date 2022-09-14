//
// Created by Mike Smith on 2022/3/23.
//

#include <core/clock.h>
#include <core/thread_pool.h>
#include <util/imageio.h>
#include <util/half.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class PlaceholderTexture final : public Texture {

private:
    float2 _uv_scale;
    float2 _uv_offset;
    uint2 _resolution;
    TextureSampler _sampler{};
    uint _channels;

public:
    PlaceholderTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc},
          _channels{std::clamp(desc->property_uint_or_default("channels", 4u), 1u, 4u)} {
        auto filter = desc->property_string_or_default("filter", "bilinear");
        auto address = desc->property_string_or_default("address", "repeat");
        for (auto &c : filter) { c = static_cast<char>(tolower(c)); }
        for (auto &c : address) { c = static_cast<char>(tolower(c)); }
        auto address_mode = [&address, desc] {
            for (auto &c : address) { c = static_cast<char>(tolower(c)); }
            if (address == "zero") { return TextureSampler::Address::ZERO; }
            if (address == "edge") { return TextureSampler::Address::EDGE; }
            if (address == "mirror") { return TextureSampler::Address::MIRROR; }
            if (address == "repeat") { return TextureSampler::Address::REPEAT; }
            LUISA_ERROR(
                "Invalid texture address mode '{}'. [{}]",
                address, desc->source_location().string());
        }();
        auto filter_mode = [&filter, desc] {
            for (auto &c : filter) { c = static_cast<char>(tolower(c)); }
            if (filter == "point") { return TextureSampler::Filter::POINT; }
            if (filter == "bilinear") { return TextureSampler::Filter::LINEAR_POINT; }
            if (filter == "trilinear") { return TextureSampler::Filter::LINEAR_LINEAR; }
            if (filter == "anisotropic" || filter == "aniso") { return TextureSampler::Filter::ANISOTROPIC; }
            LUISA_ERROR(
                "Invalid texture filter mode '{}'. [{}]",
                filter, desc->source_location().string());
        }();
        _sampler = {filter_mode, address_mode};
        _uv_scale = desc->property_float2_or_default(
            "uv_scale", lazy_construct([desc] {
                return make_float2(desc->property_float_or_default(
                    "uv_scale", 1.0f));
            }));
        _uv_offset = desc->property_float2_or_default(
            "uv_offset", lazy_construct([desc] {
                return make_float2(desc->property_float_or_default(
                    "uv_offset", 0.0f));
            }));
        _resolution = desc->property_uint2_or_default(
            "resolution", lazy_construct([desc] {
                return make_uint2(desc->property_uint_or_default(
                    "resolution", 1024u));
            }));
        _resolution = clamp(_resolution, 1u, 16384u);
        switch (semantic()) {
            case Semantic::ALBEDO: [[fallthrough]];
            case Semantic::ILLUMINANT: _channels = 4u; break;
            case Semantic::GENERIC: break;
        }
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] bool is_constant() const noexcept override { return false; }
    [[nodiscard]] auto uv_scale() const noexcept { return _uv_scale; }
    [[nodiscard]] auto uv_offset() const noexcept { return _uv_offset; }
    [[nodiscard]] auto resolution() const noexcept { return _resolution; }
    [[nodiscard]] uint channels() const noexcept override { return _channels; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PlaceholderTextureInstance final : public Texture::Instance {

private:
    const Image<float> &_image;
    luisa::optional<Differentiation::TexturedParameter> _diff_param;
    uint _texture_id{};

private:
    [[nodiscard]] Float2 _compute_uv(const Interaction &it) const noexcept {
        auto texture = node<PlaceholderTexture>();
        auto uv_scale = texture->uv_scale();
        auto uv_offset = texture->uv_offset();
        return it.uv() * uv_scale + uv_offset;
    }

public:
    PlaceholderTextureInstance(Pipeline &pipeline, CommandBuffer &command_buffer,
                               const Texture *texture, const Image<float> &image, TextureSampler sampler,
                               luisa::optional<Differentiation::TexturedParameter> param) noexcept
        : Texture::Instance{pipeline, texture},
          _image{image}, _diff_param{std::move(param)} {
        static thread_local Kernel2D fill = [](ImageFloat image) noexcept {
            image.write(dispatch_id().xy(), make_float4(.5f, .5f, .5f, 1.f));
        };
        command_buffer << fill(pipeline.device(), _image).dispatch(_image.size());
        _texture_id = pipeline.register_bindless(_image, sampler);
    }
    [[nodiscard]] Float4 evaluate(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto uv = _compute_uv(it);
        return pipeline().tex2d(_texture_id).sample(uv);// TODO: LOD
    }
    void backward(const Interaction &it, const SampledWavelengths &swl,
                  Expr<float> time, Expr<float4> grad) const noexcept override {
        if (_diff_param) {
            auto uv = _compute_uv(it);
            pipeline().differentiation()->accumulate(*_diff_param, uv, grad);
        }
    }
};

luisa::unique_ptr<Texture::Instance> PlaceholderTexture::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto storage = PixelStorage::FLOAT4;
    if (_channels == 1u) {
        storage = PixelStorage::FLOAT1;
    } else if (_channels == 2u) {
        storage = PixelStorage::FLOAT2;
    }
    auto device_image = pipeline.create<Image<float>>(storage, _resolution);
    luisa::optional<Differentiation::TexturedParameter> param;
    if (requires_gradients()) {
        param.emplace(pipeline.differentiation()->parameter(
            *device_image, _sampler, range()));
    }
    return luisa::make_unique<PlaceholderTextureInstance>(
        pipeline, command_buffer, this,
        *device_image, _sampler, std::move(param));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PlaceholderTexture)
