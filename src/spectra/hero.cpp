//
// Created by Mike Smith on 2022/3/21.
//

#include <luisa-compute.h>

#include <base/spectrum.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

static constexpr auto rsp_coefficient_scales = make_float3(10000.f, 10.f, 0.01f);
static constexpr auto inv_rsp_coefficient_scales = make_float3(1e-4f, 1e-1f, 1e2f);

class RGBSigmoidPolynomial {

private:
    Float3 _c;

private:
    [[nodiscard]] static Float _s(Expr<float> x) noexcept {
        return ite(
            isinf(x),
            cast<float>(x > 0.0f),
            0.5f + 0.5f * x * rsqrt(1.0f + x * x));
    }

public:
    RGBSigmoidPolynomial() noexcept = default;
    RGBSigmoidPolynomial(Expr<float> c0, Expr<float> c1, Expr<float> c2) noexcept
        : _c{make_float3(c0, c1, c2) * inv_rsp_coefficient_scales} {}
    explicit RGBSigmoidPolynomial(Expr<float3> c) noexcept
        : _c{c * inv_rsp_coefficient_scales} {}
    [[nodiscard]] Float operator()(Expr<float> lambda) const noexcept {
        return _s(fma(lambda, fma(lambda, _c.x, _c.y), _c.z));// c0 * x * x + c1 * x + c2
    }
    [[nodiscard]] Float maximum() const noexcept {
        auto edge = max(
            (*this)(visible_wavelength_min),
            (*this)(visible_wavelength_max));
        auto mid = (*this)(clamp(
            -_c.y / (2.0f * _c.x),
            visible_wavelength_min,
            visible_wavelength_max));
        return max(edge, mid);
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
    [[nodiscard]] RGBSigmoidPolynomial decode_albedo(
        const BindlessArray &array, Expr<uint> base_index, Expr<float3> rgb_in) const noexcept {
        static Callable decode = [](BindlessVar array, UInt base_index, Float3 rgb_in) noexcept {
            auto rgb = clamp(rgb_in, 0.0f, 1.0f);
            auto c = rsp_coefficient_scales *
                     make_float3(0.0f, 0.0f, (rgb[0] - 0.5f) / sqrt(rgb[0] * (1.0f - rgb[0])));
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
        return RGBSigmoidPolynomial{decode(Expr{array}, base_index, rgb_in)};
    }
    [[nodiscard]] std::pair<RGBSigmoidPolynomial, Float> decode_unbound(
        const BindlessArray &array, Expr<uint> base_index, Expr<float3> rgb) const noexcept {
        auto m = max(max(rgb.x, rgb.y), rgb.z);
        auto scale = 2.0f * m;
        auto c = decode_albedo(
            array, base_index,
            ite(scale == 0.0f, 0.0f, rgb / scale));
        return std::make_pair(std::move(c), std::move(scale));
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
    const DenselySampledSpectrum *_illuminant;

public:
    RGBIlluminantSpectrum(RGBSigmoidPolynomial rsp, Expr<float> scale, const DenselySampledSpectrum &illum) noexcept
        : _rsp{std::move(rsp)}, _scale{scale}, _illuminant{&illum} {}
    [[nodiscard]] auto sample(Expr<float> lambda) const noexcept {
        return _rsp(lambda) * _scale * _illuminant->sample(lambda);
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
};

class HeroWavelengthSpectrumInstance final : public Spectrum::Instance {

private:
    uint _rgb2spec_t0;

public:
    HeroWavelengthSpectrumInstance(
        const Pipeline &pipeline, const Spectrum *spectrum, uint t0) noexcept
        : Spectrum::Instance{pipeline, spectrum}, _rgb2spec_t0{t0} {}
    [[nodiscard]] SampledSpectrum albedo_from_srgb(
        const SampledWavelengths &swl, Expr<float3> rgb) const noexcept override {
        auto rsp = RGB2SpectrumTable::srgb().decode_albedo(
            pipeline().bindless_array(), _rgb2spec_t0, rgb);
        auto spec = RGBAlbedoSpectrum{std::move(rsp)};
        SampledSpectrum s{node()->dimension()};
        for (auto i = 0u; i < s.dimension(); i++) {
            s[i] = spec.sample(swl.lambda(i));
        }
        return s;
    }
    [[nodiscard]] SampledSpectrum illuminant_from_srgb(
        const SampledWavelengths &swl, Expr<float3> rgb_in) const noexcept override {
        auto rgb = max(rgb_in, 0.f);
        auto [rsp, scale] = RGB2SpectrumTable::srgb().decode_unbound(
            pipeline().bindless_array(), _rgb2spec_t0, rgb);
        auto spec = RGBIlluminantSpectrum{
            std::move(rsp), std::move(scale),
            DenselySampledSpectrum::cie_illum_d65()};
        SampledSpectrum s{node()->dimension()};
        for (auto i = 0u; i < s.dimension(); i++) {
            s[i] = spec.sample(swl.lambda(i));
        }
        return s;
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
    RGB2SpectrumTable::srgb().encode(
        command_buffer,
        rgb2spec_t0->view(0u),
        rgb2spec_t1->view(0u),
        rgb2spec_t2->view(0u));
    auto t0 = pipeline.register_bindless(*rgb2spec_t0, TextureSampler::linear_point_zero());
    auto t1 = pipeline.register_bindless(*rgb2spec_t1, TextureSampler::linear_point_zero());
    auto t2 = pipeline.register_bindless(*rgb2spec_t2, TextureSampler::linear_point_zero());
    LUISA_ASSERT(
        t1 == t0 + 1u && t2 == t0 + 2u,
        "Invalid RGB2Spec texture indices: "
        "{}, {}, and {}.",
        t0, t1, t2);
    return luisa::make_unique<HeroWavelengthSpectrumInstance>(pipeline, this, t0);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HeroWavelengthSpectrum)
