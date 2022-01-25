//
// Created by Mike Smith on 2022/1/19.
//

#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <util/colorspace.h>
#include <util/spectrum.h>

namespace luisa::render {

inline Float RGBSigmoidPolynomial::_s(Expr<float> x) noexcept {
    using namespace luisa::compute;
    return ite(
        isinf(x),
        cast<float>(x > 0.0f),
        0.5f + 0.5f * x * rsqrt(1.0f + x * x));
}

Float4 RGBSigmoidPolynomial::_s(Expr<float4> x) noexcept {
    using namespace luisa::compute;
    return ite(
        isinf(x),
        make_float4(x > 0.0f),
        0.5f + 0.5f * x * rsqrt(1.0f + x * x));
}

Float RGBSigmoidPolynomial::operator()(Expr<float> lambda) const noexcept {
    using luisa::compute::fma;
    return _s(fma(lambda, fma(lambda, _c0, _c1), _c2));// c0 * x * x + c1 * x + c2
}

Float4 RGBSigmoidPolynomial::operator()(Expr<float4> lambda) const noexcept {
    using luisa::compute::fma;
    return _s(fma(lambda, fma(lambda, _c0, _c1), _c2));// c0 * x * x + c1 * x + c2
}

Float RGBSigmoidPolynomial::maximum() const noexcept {
    using namespace luisa::compute;
    auto edge = max(
        (*this)(visible_wavelength_min),
        (*this)(visible_wavelength_max));
    auto mid = (*this)(clamp(
        -_c1 / (2.0f * _c0),
        visible_wavelength_min,
        visible_wavelength_max));
    return max(edge, mid);
}

Bool SampledWavelengths::operator==(const SampledWavelengths &rhs) const noexcept {
    return all(_lambda == rhs._lambda) & all(_pdf == rhs._pdf);
}

Bool SampledWavelengths::operator!=(const SampledWavelengths &rhs) const noexcept {
    return any(_lambda != rhs._lambda) | any(_pdf != rhs._pdf);
}

SampledWavelengths SampledWavelengths::sample_uniform(Expr<float> u, Expr<float> lambda_min, Expr<float> lambda_max) noexcept {
    using namespace luisa::compute;
    auto l = lambda_max - lambda_min;
    auto delta = l * (1.0f / sample_count);
    auto primary = lerp(u, lambda_min, lambda_max);
    auto secondary = primary + delta * make_float3(1.0f, 2.0f, 3.0f);
    secondary = ite(secondary <= lambda_max, secondary, secondary - l);
    return {make_float4(1.0f / l), make_float4(primary, secondary)};
}

SampledWavelengths SampledWavelengths::sample_visible(Expr<float> u) noexcept {
    constexpr auto sample_visible_wavelengths = [](auto u) noexcept {
        using luisa::compute::atanh;
        return 538.0f - 138.888889f * atanh(0.85691062f - 1.82750197f * u);
    };
    constexpr auto visible_wavelengths_pdf = [](auto lambda) noexcept {
        using luisa::compute::ite;
        using luisa::compute::cosh;
        constexpr auto sqr = [](auto x) noexcept { return x * x; };
        return ite(
            lambda >= visible_wavelength_min &&
                lambda <= visible_wavelength_max,
            0.0039398042f / sqr(cosh(0.0072f * (lambda - 538.0f))),
            0.0f);
    };
    using luisa::compute::fract;
    auto offset = make_float4(0.0f, 1.0f, 2.0f, 3.0f) * (1.0f / sample_count);
    auto up = fract(u + offset);
    auto lambda = sample_visible_wavelengths(up);
    auto pdf = visible_wavelengths_pdf(lambda);
    return {lambda, pdf};
}

Bool SampledWavelengths::secondary_terminated() const noexcept {
    using luisa::compute::all;
    return all(_pdf == 0.0f);
}

void SampledWavelengths::terminate_secondary() noexcept {
    using luisa::compute::ite;
    using luisa::compute::make_float4;
    _pdf = ite(
        secondary_terminated(),
        _pdf,
        make_float4(
            _pdf.x * (1.0f / sample_count),
            0.0f, 0.0f, 0.0f));
}

#include <util/spectrum_cie_xyz.inl.h>
#include <util/spectrum_cie_illum_d65.inl.h>

const DenselySampledSpectrum &DenselySampledSpectrum::cie_x() noexcept {
    static DenselySampledSpectrum s{cie_x_samples};
    return s;
}

const DenselySampledSpectrum &DenselySampledSpectrum::cie_y() noexcept {
    static DenselySampledSpectrum s{cie_y_samples};
    return s;
}

const DenselySampledSpectrum &DenselySampledSpectrum::cie_z() noexcept {
    static DenselySampledSpectrum s{cie_z_samples};
    return s;
}

const DenselySampledSpectrum &DenselySampledSpectrum::cie_illum_d6500() noexcept {
    static DenselySampledSpectrum s{cie_illum_d6500_samples};
    return s;
}

luisa::compute::Float4 DenselySampledSpectrum::sample(const SampledWavelengths &swl) const noexcept {
    using namespace luisa::compute;
    auto lambda = swl.lambda();
    auto t = lambda - visible_wavelength_min;
    auto i = make_uint4(clamp(t, 0.0f, static_cast<float>(cie_sample_count - 2u)));
    auto s0 = make_float4(_values[i.x], _values[i.y], _values[i.z], _values[i.w]);
    auto s1 = make_float4(_values[i.x + 1u], _values[i.y + 1u], _values[i.z + 1u], _values[i.w + 1u]);
    auto w = t - make_float4(i);
    return ite(
        lambda >= visible_wavelength_min &&
            lambda <= visible_wavelength_max,
        lerp(s0, s1, w),
        make_float4(0.0f));
}

Float SampledWavelengths::cie_y(Expr<float4> values) const noexcept {
    auto y = DenselySampledSpectrum::cie_y().sample(*this);
    constexpr auto average = [](auto v) noexcept {
        return (v.x + v.y + v.z + v.w) *
               (1.0f / SampledWavelengths::sample_count);
    };
    constexpr auto safe_div = [](auto a, auto b) noexcept {
        using luisa::compute::ite;
        return ite(b == 0.0f, 0.0f, a / b);
    };
    using luisa::compute::make_float3;
    return average(safe_div(y * values, _pdf)) *
           inv_cie_y_integral;
}

Float3 SampledWavelengths::cie_xyz(Expr<float4> values) const noexcept {
    auto x = DenselySampledSpectrum::cie_x().sample(*this);
    auto y = DenselySampledSpectrum::cie_y().sample(*this);
    auto z = DenselySampledSpectrum::cie_z().sample(*this);
    constexpr auto average = [](auto v) noexcept {
        return (v.x + v.y + v.z + v.w) *
               (1.0f / SampledWavelengths::sample_count);
    };
    constexpr auto safe_div = [](auto a, auto b) noexcept {
        using luisa::compute::ite;
        return ite(b == 0.0f, 0.0f, a / b);
    };
    using luisa::compute::make_float3;
    return make_float3(
               average(safe_div(x * values, _pdf)),
               average(safe_div(y * values, _pdf)),
               average(safe_div(z * values, _pdf))) *
           inv_cie_y_integral;
}

Float3 SampledWavelengths::srgb(Expr<float4> values) const noexcept {
    return cie_xyz_to_linear_srgb(cie_xyz(values));
}

extern "C" RGB2SpectrumTable::scale_table_type sRGBToSpectrumTable_Scale;
extern "C" RGB2SpectrumTable::coefficient_table_type sRGBToSpectrumTable_Data;

RGB2SpectrumTable RGB2SpectrumTable::srgb() noexcept {
    return {sRGBToSpectrumTable_Scale, sRGBToSpectrumTable_Data};
}

void RGB2SpectrumTable::encode(CommandBuffer &command_buffer, VolumeView<float> t0, VolumeView<float> t1, VolumeView<float> t2) const noexcept {
    command_buffer << t0.copy_from(_coefficients[0])
                   << t1.copy_from(_coefficients[1])
                   << t2.copy_from(_coefficients[2])
                   << luisa::compute::commit();
}

// from PBRT-v4: https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/color.cpp
float3 RGB2SpectrumTable::decode_albedo(float3 rgb_in) const noexcept {
    auto rgb = clamp(rgb_in, 0.0f, 1.0f);
    if (rgb[0] == rgb[1] && rgb[1] == rgb[2]) {
        return make_float3(
            0.0f, 0.0f, (rgb[0] - 0.5f) / std::sqrt(rgb[0] * (1.0f - rgb[0])));
    }

    // Find maximum component and compute remapped component values
    auto maxc = (rgb[0] > rgb[1]) ?
                    ((rgb[0] > rgb[2]) ? 0u : 2u) :
                    ((rgb[1] > rgb[2]) ? 1u : 2u);
    auto z = rgb[maxc];
    auto x = rgb[(maxc + 1u) % 3u] * (resolution - 1u) / z;
    auto y = rgb[(maxc + 2u) % 3u] * (resolution - 1u) / z;

    // Compute integer indices and offsets for coefficient interpolation
    auto xi = std::min(static_cast<uint>(x), resolution - 2u);
    auto yi = std::min(static_cast<uint>(y), resolution - 2u);
    auto zi = std::min(
        static_cast<uint>(std::lower_bound(_z_nodes, _z_nodes + resolution, z) - _z_nodes),
        resolution - 2u);
    auto dx = x - static_cast<float>(xi);
    auto dy = y - static_cast<float>(yi);
    auto dz = (z - _z_nodes[zi]) / (_z_nodes[zi + 1u] - _z_nodes[zi]);

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

// FIXME: producing monochrome images...
RGBSigmoidPolynomial RGB2SpectrumTable::decode_albedo(Expr<BindlessArray> array, Expr<uint> base_index, Expr<float3> rgb_in) const noexcept {
    using namespace luisa::compute;
    auto rgb = clamp(rgb_in, 0.0f, 1.0f);
    auto c = make_float3(
        0.0f, 0.0f, (rgb[0] - 0.5f) / sqrt(rgb[0] * (1.0f - rgb[0])));
    $if(rgb[0] != rgb[1] | rgb[1] != rgb[2]) {
        // Find maximum component and compute remapped component values
        auto maxc = ite(
            rgb[0] > rgb[1],
            ite(rgb[0] > rgb[2], 0u, 2u),
            ite(rgb[1] > rgb[2], 1u, 2u));
        auto z = rgb[maxc];
        auto x = rgb[(maxc + 1u) % 3u] * (resolution - 1u) / z;
        auto y = rgb[(maxc + 2u) % 3u] * (resolution - 1u) / z;

        // Compute integer indices and offsets for coefficient interpolation
        Constant<float> z_nodes{_z_nodes};
        auto find_z_interval = [&z_nodes](auto z) noexcept {
            auto size = def(resolution - 2u);
            auto first = def(1u);
            $while(size != 0u) {
                auto half = size >> 1u;
                auto middle = first + half;
                auto pred = z_nodes[middle] < z;
                first = ite(pred, middle + 1u, first);
                size = ite(pred, size - (half + 1u), half);
            };
            return min(first - 1u, resolution - 2u);
        };
        auto zi = find_z_interval(z);
        auto dz = (z - z_nodes[zi]) / (z_nodes[zi + 1u] - z_nodes[zi]);

        // Trilinearly interpolate sigmoid polynomial coefficients _c_
        auto coord = make_float3(x, y, cast<float>(zi) + dz) + 0.5f;
        c = array.tex3d(base_index + maxc)
                .sample(coord * (1.0f / resolution))
                .xyz();
    };
    return RGBSigmoidPolynomial{c};
}

std::pair<float3, float> RGB2SpectrumTable::decode_unbound(float3 rgb) const noexcept {
    auto m = std::max({rgb.x, rgb.y, rgb.z});
    auto scale = 2.0f * m;
    auto c = decode_albedo(scale == 0.0f ? make_float3(0.0f) : rgb / scale);
    return std::make_pair(c, scale);
}

std::pair<RGBSigmoidPolynomial, Float> RGB2SpectrumTable::decode_unbound(
    Expr<BindlessArray> array, Expr<uint> base_index, Expr<float3> rgb) const noexcept {
    using namespace luisa::compute;
    auto m = max(max(rgb.x, rgb.y), rgb.z);
    auto scale = 2.0f * m;
    auto c = decode_albedo(
        array, base_index,
        ite(scale == 0.0f, 0.0f, rgb / scale));
    return std::make_pair(std::move(c), std::move(scale));
}

}// namespace luisa::render
