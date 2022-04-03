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

class ImageTexture final : public Texture {

public:
    enum struct Encoding : uint {
        LINEAR,
        SRGB,
        GAMMA,
    };

private:
    std::shared_future<LoadedImage> _image;
    float2 _uv_scale;
    float2 _uv_offset;
    TextureSampler _sampler{};
    Encoding _encoding{};
    float _scale{1.f};
    float _gamma{1.f};

private:
    void _load_image(const SceneNodeDesc *desc) noexcept {
        auto path = desc->property_path("file");
        auto encoding = desc->property_string_or_default(
            "encoding", lazy_construct([&path]() noexcept -> luisa::string {
                auto ext = path.extension().string();
                for (auto &c : ext) { c = static_cast<char>(tolower(c)); }
                if (ext == ".exr" || ext == ".hdr") { return "linear"; }
                return "sRGB";
            }));
        for (auto &c : encoding) { c = static_cast<char>(tolower(c)); }
        if (encoding == "srgb") {
            _encoding = Encoding::SRGB;
        } else if (encoding == "gamma") {
            _encoding = Encoding::GAMMA;
            _gamma = desc->property_float_or_default("gamma", 1.f);
        } else {
            if (encoding != "linear") [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Unknown texture encoding '{}'. "
                    "Fallback to linear encoding. [{}]",
                    encoding, desc->source_location().string());
            }
            _encoding = Encoding::LINEAR;
        }
        _scale = desc->property_float_or_default("scale", 1.f);
        _image = ThreadPool::global().async([this, path = std::move(path)] {
            if (requires_gradients()) {
                auto storage = LoadedImage::parse_storage(path);
                auto channels = pixel_storage_channel_count(storage);
                auto image = [&] {
                    if (channels == 1u) { return LoadedImage::load(path, PixelStorage::FLOAT1); }
                    if (channels == 2u) { return LoadedImage::load(path, PixelStorage::FLOAT2); }
                    return LoadedImage::load(path, PixelStorage::FLOAT4);
                }();
                auto pixels = static_cast<float *>(image.pixels());
                auto n = image.pixel_count() * image.channels();
                if (_encoding == Encoding::SRGB) {
                    for (auto i = 0u; i < n; i++) {
                        auto x = pixels[i];
                        auto p = x <= 0.04045f ?
                                     x * (1.f / 12.92f) :
                                     std::pow((x + 0.055f) * (1.0f / 1.055f), 2.4f);
                        pixels[i] = _scale * p;
                    }
                } else if (_encoding == Encoding::GAMMA) {
                    for (auto i = 0u; i < n; i++) {
                        pixels[i] = _scale * pow(pixels[i], _gamma);
                    }
                } else {// _encoding == Encoding::LINEAR
                    for (auto i = 0u; i < n; i++) {
                        pixels[i] *= _scale;
                    }
                }
                return image;
            }
            return LoadedImage::load(path);
        });
    }

public:
    ImageTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} {
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
        _load_image(desc);
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.f; }
    [[nodiscard]] bool is_constant() const noexcept override { return false; }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto gamma() const noexcept { return _gamma; }
    [[nodiscard]] auto uv_scale() const noexcept { return _uv_scale; }
    [[nodiscard]] auto uv_offset() const noexcept { return _uv_offset; }
    [[nodiscard]] auto encoding() const noexcept { return _encoding; }
    [[nodiscard]] uint channels() const noexcept override { return _image.get().channels(); }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class ImageTextureInstance final : public Texture::Instance {

private:
    luisa::optional<Differentiation::TexturedParameter> _diff_param;
    uint _texture_id;

private:
    [[nodiscard]] Float2 _compute_uv(const Interaction &it) const noexcept {
        auto texture = node<ImageTexture>();
        auto uv_scale = texture->uv_scale();
        auto uv_offset = texture->uv_offset();
        return it.uv() * uv_scale + uv_offset;
    }

    [[nodiscard]] Float4 _decode(Expr<float4> rgba) const noexcept {
        if (_diff_param) { return rgba; }// already pre-processed
        auto texture = node<ImageTexture>();
        auto encoding = texture->encoding();
        auto scale = texture->scale();
        if (encoding == ImageTexture::Encoding::SRGB) {
            auto linear = ite(
                rgba <= 0.04045f,
                rgba * (1.0f / 12.92f),
                pow((rgba + 0.055f) * (1.0f / 1.055f), 2.4f));
            return make_float4(scale * linear);
        }
        if (encoding == ImageTexture::Encoding::GAMMA) {
            auto gamma = texture->gamma();
            return scale * pow(rgba, gamma);
        }
        return scale * rgba;
    }

public:
    ImageTextureInstance(const Pipeline &pipeline, const Texture *texture, uint texture_id,
                         luisa::optional<Differentiation::TexturedParameter> param) noexcept
        : Texture::Instance{pipeline, texture},
          _texture_id{texture_id}, _diff_param{std::move(param)} {}
    [[nodiscard]] Float4 evaluate(const Interaction &it, Expr<float> time) const noexcept override {
        auto uv = _compute_uv(it);
        auto v = pipeline().tex2d(_texture_id).sample(uv);// TODO: LOD
        return _decode(v);
    }
    void backward(const Interaction &it, Expr<float> time, Expr<float4> grad) const noexcept override {
        if (_diff_param) {
            auto uv = _compute_uv(it);
            pipeline().differentiation().accumulate(*_diff_param, uv, grad);
        }
    }
};

luisa::unique_ptr<Texture::Instance> ImageTexture::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto &&image = _image.get();
    auto device_image = pipeline.create<Image<float>>(image.pixel_storage(), image.size());
    auto tex_id = pipeline.register_bindless(*device_image, _sampler);
    command_buffer << device_image->copy_from(image.pixels())
                   << compute::commit();
    luisa::optional<Differentiation::TexturedParameter> param;
    if (requires_gradients()) {
        param.emplace(pipeline.differentiation().parameter(
            *device_image, _sampler, range()));
    }
    return luisa::make_unique<ImageTextureInstance>(
        pipeline, this, tex_id, std::move(param));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ImageTexture)
