//
// Created by Mike Smith on 2022/3/21.
//

#include <luisa-compute.h>
#include <base/spectrum.h>

namespace luisa::render {

using namespace compute;

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
    [[nodiscard]] float3 decode_albedo(float3 rgb_in) const noexcept {
        auto rgb = clamp(rgb_in, 0.0f, 1.0f);
        if (rgb[0] == rgb[1] && rgb[1] == rgb[2]) {
            return rsp_coefficient_scales *
                   make_float3(0.0f, 0.0f, (rgb[0] - 0.5f) / std::sqrt(rgb[0] * (1.0f - rgb[0])));
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
        return c;
    }
    [[nodiscard]] std::pair<float3, float> decode_unbound(float3 rgb) const noexcept {
        auto m = std::max({rgb.x, rgb.y, rgb.z});
        auto scale = 2.0f * m;
        auto c = decode_albedo(scale == 0.0f ? make_float3(0.0f) : rgb / scale);
        return std::make_pair(c, scale);
    }
    [[nodiscard]] RGBSigmoidPolynomial decode_albedo(Expr<BindlessArray> array, Expr<uint> base_index, Expr<float3> rgb_in) const noexcept {
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
        return RGBSigmoidPolynomial{decode(array, base_index, rgb_in)};
    }
    [[nodiscard]] std::pair<RGBSigmoidPolynomial, Float> decode_unbound(Expr<BindlessArray> array, Expr<uint> base_index, Expr<float3> rgb) const noexcept {
        auto m = max(max(rgb.x, rgb.y), rgb.z);
        auto scale = 2.0f * m;
        auto c = decode_albedo(
            array, base_index,
            ite(scale == 0.0f, 0.0f, rgb / scale));
        return std::make_pair(std::move(c), std::move(scale));
    }
    void encode(CommandBuffer &command_buffer, VolumeView<float> t0, VolumeView<float> t1, VolumeView<float> t2) const noexcept {
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

}// namespace luisa::render
