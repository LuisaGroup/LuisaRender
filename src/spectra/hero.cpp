//
// Created by Mike Smith on 2022/3/21.
//

#include <luisa-compute.h>
#include <base/spd.h>
#include <base/spectrum.h>
#include <base/pipeline.h>

namespace luisa::render {

using luisa::select;
using std::max;
using std::sin;
using namespace luisa::compute;

class RGBSigmoidPolynomial {

private:
    Float3 _c;

private:
    [[nodiscard]] static Float _s(Expr<float> x) noexcept {
        return ite(isinf(x), cast<float>(x > 0.0f),
                   0.5f * fma(x, rsqrt(fma(x, x, 1.f)), 1.f));
    }

public:
    RGBSigmoidPolynomial() noexcept = default;
    RGBSigmoidPolynomial(Expr<float> c0, Expr<float> c1, Expr<float> c2) noexcept
        : _c{make_float3(c0, c1, c2)} {}
    explicit RGBSigmoidPolynomial(Expr<float3> c) noexcept : _c{c} {}
    [[nodiscard]] Float operator()(Expr<float> lambda) const noexcept {
        return _s(fma(lambda, fma(lambda, _c.x, _c.y), _c.z));// c0 * x * x + c1 * x + c2
    }
};

class RGB2SpectrumTable {

public:
    static constexpr auto resolution = 64u;
    using coefficient_table_type = const float[3][resolution][resolution][resolution][4];

private:
    const coefficient_table_type &_coefficients;

private:
    [[nodiscard]] inline static auto _inverse_smooth_step(auto x) noexcept {
        return 0.5f - sin(asin(1.0f - 2.0f * x) * (1.0f / 3.0f));
    }

public:
    explicit constexpr RGB2SpectrumTable(const coefficient_table_type &coefficients) noexcept
        : _coefficients{coefficients} {}
    constexpr RGB2SpectrumTable(RGB2SpectrumTable &&) noexcept = default;
    constexpr RGB2SpectrumTable(const RGB2SpectrumTable &) noexcept = default;
    [[nodiscard]] static RGB2SpectrumTable srgb() noexcept;
    [[nodiscard]] Float4 decode_albedo(
        const BindlessArray &array, Expr<uint> base_index, Expr<float3> rgb_in) const noexcept {
        auto rgb = clamp(rgb_in, 0.0f, 1.0f);
        static Callable decode = [](BindlessVar array, UInt base_index, Float3 rgb) noexcept {
            auto c = make_float3(0.0f, 0.0f, (rgb[0] - 0.5f) * rsqrt(rgb[0] * (1.0f - rgb[0])));
            $if(rgb[0] != rgb[1] | rgb[1] != rgb[2]) {
                // Find maximum component and compute remapped component values
                auto maxc = ite(
                    rgb[0] > rgb[1],
                    ite(rgb[0] > rgb[2], 0u, 2u),
                    ite(rgb[1] > rgb[2], 1u, 2u));
                auto z = rgb[maxc];
                auto x = rgb[(maxc + 1u) % 3u] / z;
                auto y = rgb[(maxc + 2u) % 3u] / z;
                auto zz = _inverse_smooth_step(_inverse_smooth_step(z));

                // Trilinearly interpolate sigmoid polynomial coefficients _c_
                auto coord = fma(
                    make_float3(x, y, zz),
                    (resolution - 1.0f) / resolution,
                    0.5f / resolution);
                c = array.tex3d(base_index + maxc).sample(coord).xyz();
            };
            return c;
        };
        return make_float4(decode(Expr{array}, base_index, rgb), srgb_to_cie_y(rgb));
    }

    [[nodiscard]] float4 decode_albedo(float3 rgb_in) const noexcept {
        auto rgb = clamp(rgb_in, 0.0f, 1.0f);
        if (rgb[0] == rgb[1] && rgb[1] == rgb[2]) {
            auto s = (rgb[0] - 0.5f) / std::sqrt(rgb[0] * (1.0f - rgb[0]));
            return make_float4(0.0f, 0.0f, s, srgb_to_cie_y(rgb));
        }
        // Find maximum component and compute remapped component values
        auto maxc = (rgb[0] > rgb[1]) ?
                        ((rgb[0] > rgb[2]) ? 0u : 2u) :
                        ((rgb[1] > rgb[2]) ? 1u : 2u);
        auto z = rgb[maxc];
        auto x = rgb[(maxc + 1u) % 3u] * (resolution - 1u) / z;
        auto y = rgb[(maxc + 2u) % 3u] * (resolution - 1u) / z;

        auto zz = _inverse_smooth_step(_inverse_smooth_step(z)) * (resolution - 1u);

        // Compute integer indices and offsets for coefficient interpolation
        auto xi = std::min(static_cast<uint>(x), resolution - 2u);
        auto yi = std::min(static_cast<uint>(y), resolution - 2u);
        auto zi = std::min(static_cast<uint>(zz), resolution - 2u);
        auto dx = x - static_cast<float>(xi);
        auto dy = y - static_cast<float>(yi);
        auto dz = zz - static_cast<float>(zi);

        // Trilinearly interpolate sigmoid polynomial coefficients _c_
        auto c = make_float3();
        for (auto i = 0u; i < 3u; i++) {
            // Define _co_ lambda for looking up sigmoid polynomial coefficients
            auto co = [=, this](int dx, int dy, int dz) noexcept {
                return _coefficients[maxc][zi + dz][yi + dy][xi + dx][i];
            };
            c[i] = lerp(lerp(lerp(co(0, 0, 0), co(1, 0, 0), dx),
                             lerp(co(0, 1, 0), co(1, 1, 0), dx), dy),
                        lerp(lerp(co(0, 0, 1), co(1, 0, 1), dx),
                             lerp(co(0, 1, 1), co(1, 1, 1), dx), dy),
                        dz);
        }
        return make_float4(c, srgb_to_cie_y(rgb));
    }

    [[nodiscard]] Float4 decode_unbound(
        const BindlessArray &array, Expr<uint> base_index, Expr<float3> rgb_in) const noexcept {
        auto rgb = max(rgb_in, 0.0f);
        auto m = max(max(rgb.x, rgb.y), rgb.z);
        auto scale = 2.0f * m;
        auto c = decode_albedo(
            array, base_index,
            ite(scale == 0.0f, 0.0f, rgb / scale));
        return make_float4(c.xyz(), scale);
    }

    [[nodiscard]] float4 decode_unbound(float3 rgb) const noexcept {
        auto m = max(max(rgb.x, rgb.y), rgb.z);
        auto scale = 2.0f * m;
        auto c = decode_albedo(scale == 0.f ? make_float3() : rgb / scale);
        return make_float4(c.xyz(), scale);
    }

    void encode(CommandBuffer &command_buffer,
                VolumeView<float> t0, VolumeView<float> t1, VolumeView<float> t2) const noexcept {
        command_buffer << t0.copy_from(_coefficients[0])
                       << t1.copy_from(_coefficients[1])
                       << t2.copy_from(_coefficients[2])
                       << luisa::compute::commit();
    }
};

extern RGB2SpectrumTable::coefficient_table_type sRGBToSpectrumTable_Data;

RGB2SpectrumTable RGB2SpectrumTable::srgb() noexcept {
    return RGB2SpectrumTable{sRGBToSpectrumTable_Data};
}

class RGBAlbedoSpectrum {

private:
    RGBSigmoidPolynomial _rsp;

public:
    explicit RGBAlbedoSpectrum(RGBSigmoidPolynomial rsp) noexcept : _rsp{std::move(rsp)} {}
    [[nodiscard]] auto sample(Expr<float> lambda) const noexcept { return _rsp(lambda); }
};

class RGBIlluminantSpectrum {

private:
    RGBSigmoidPolynomial _rsp;
    Float _scale;
    SPD _illum;

public:
    RGBIlluminantSpectrum(RGBSigmoidPolynomial rsp, Expr<float> scale, SPD illum) noexcept
        : _rsp{std::move(rsp)}, _scale{scale}, _illum{illum} {}
    [[nodiscard]] auto sample(Expr<float> lambda) const noexcept {
        return _rsp(lambda) * _scale * _illum.sample(lambda);
    }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
};

class HeroWavelengthSpectrum final : public Spectrum {

private:
    uint _dimension{};

public:
    HeroWavelengthSpectrum(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Spectrum{scene, desc},
          _dimension{std::max(desc->property_uint_or_default("dimension", 4u), 1u)} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_differentiable() const noexcept override { return false; }
    [[nodiscard]] bool is_fixed() const noexcept override { return false; }
    [[nodiscard]] uint dimension() const noexcept override { return _dimension; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] bool requires_encoding() const noexcept override { return true; }
    [[nodiscard]] float4 encode_srgb_albedo(float3 rgb) const noexcept override {
        return RGB2SpectrumTable::srgb().decode_albedo(rgb);
    }
    [[nodiscard]] float4 encode_srgb_illuminant(float3 rgb) const noexcept override {
        return RGB2SpectrumTable::srgb().decode_unbound(rgb);
    }
    [[nodiscard]] PixelStorage encoded_albedo_storage(PixelStorage storage) const noexcept override {
        LUISA_ASSERT(pixel_storage_channel_count(storage) != 2u,
                     "Hero-wavelength spectrum does not support 2-channel albedo pixels.");
        return storage == PixelStorage::FLOAT1 || storage == PixelStorage::FLOAT4 ?
                   PixelStorage::FLOAT4 :
                   PixelStorage::HALF4;
    }
    [[nodiscard]] PixelStorage encoded_illuminant_storage(PixelStorage storage) const noexcept override {
        LUISA_ASSERT(pixel_storage_channel_count(storage) != 2u,
                     "Hero-wavelength spectrum does not support 2-channel illuminant pixels.");
        return storage == PixelStorage::FLOAT1 || storage == PixelStorage::FLOAT4 ?
                   PixelStorage::FLOAT4 :
                   PixelStorage::HALF4;
    }
};

class HeroWavelengthSpectrumInstance final : public Spectrum::Instance {

private:
    SPD _illum_d65;
    uint _rgb2spec_t0;

public:
    HeroWavelengthSpectrumInstance(Pipeline &pipeline, CommandBuffer &cb,
                                   const Spectrum *spectrum, uint t0) noexcept
        : Spectrum::Instance{pipeline, cb, spectrum},
          _illum_d65{SPD::create_cie_d65(pipeline, cb)},
          _rgb2spec_t0{t0} {}
    [[nodiscard]] Spectrum::Decode decode_albedo(
        const SampledWavelengths &swl, Expr<float4> v) const noexcept override {
        auto spec = RGBAlbedoSpectrum{RGBSigmoidPolynomial{v.xyz()}};
        SampledSpectrum s{node()->dimension()};
        for (auto i = 0u; i < s.dimension(); i++) {
            s[i] = spec.sample(swl.lambda(i));
        }
        return {.value = s, .strength = v.w};
    }
    [[nodiscard]] Spectrum::Decode decode_illuminant(
        const SampledWavelengths &swl, Expr<float4> v) const noexcept override {
        auto spec = RGBIlluminantSpectrum{
            RGBSigmoidPolynomial{v.xyz()}, v.w, _illum_d65};
        SampledSpectrum s{node()->dimension()};
        for (auto i = 0u; i < s.dimension(); i++) {
            s[i] = spec.sample(swl.lambda(i));
        }
        return {.value = s, .strength = v.w};
    }
    [[nodiscard]] Float4 encode_srgb_albedo(Expr<float3> rgb) const noexcept override {
        return RGB2SpectrumTable::srgb().decode_albedo(pipeline().bindless_array(), _rgb2spec_t0, rgb);
    }
    [[nodiscard]] Float4 encode_srgb_illuminant(Expr<float3> rgb) const noexcept override {
        return RGB2SpectrumTable::srgb().decode_unbound(pipeline().bindless_array(), _rgb2spec_t0, rgb);
    }
};

luisa::unique_ptr<Spectrum::Instance> HeroWavelengthSpectrum::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto rgb2spec_t0 = pipeline.create<Volume<float>>(
        PixelStorage::FLOAT4, make_uint3(RGB2SpectrumTable::resolution));
    auto rgb2spec_t1 = pipeline.create<Volume<float>>(
        PixelStorage::FLOAT4, make_uint3(RGB2SpectrumTable::resolution));
    auto rgb2spec_t2 = pipeline.create<Volume<float>>(
        PixelStorage::FLOAT4, make_uint3(RGB2SpectrumTable::resolution));
    RGB2SpectrumTable::srgb().encode(command_buffer, *rgb2spec_t0, *rgb2spec_t1, *rgb2spec_t2);
    auto t0 = pipeline.register_bindless(*rgb2spec_t0, TextureSampler::linear_point_zero());
    auto t1 = pipeline.register_bindless(*rgb2spec_t1, TextureSampler::linear_point_zero());
    auto t2 = pipeline.register_bindless(*rgb2spec_t2, TextureSampler::linear_point_zero());
    LUISA_ASSERT(
        t1 == t0 + 1u && t2 == t0 + 2u,
        "Invalid RGB2Spec texture indices: "
        "{}, {}, and {}.",
        t0, t1, t2);
    return luisa::make_unique<HeroWavelengthSpectrumInstance>(
        pipeline, command_buffer, this, t0);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HeroWavelengthSpectrum)
