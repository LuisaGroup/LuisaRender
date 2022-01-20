//
// Created by Mike Smith on 2022/1/19.
//

#include <dsl/syntax.h>
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

Float RGBSigmoidPolynomial::operator()(Expr<float> lambda) const noexcept {
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

Float sampled_spectrum_to_y(const SampledWavelengths &lambda, Expr<float4> values) noexcept {
    auto y = sample_cie_y(lambda);
    constexpr auto average = [](auto v) noexcept {
        return (v.x + v.y + v.z + v.w) *
               (1.0f / SampledWavelengths::sample_count);
    };
    constexpr auto safe_div = [](auto a, auto b) noexcept {
        using luisa::compute::ite;
        return ite(b == 0.0f, 0.0f, a / b);
    };
    using luisa::compute::make_float3;
    return average(safe_div(y * values, lambda.pdf()));
}

Float3 sampled_spectrum_to_xyz(const SampledWavelengths &lambda, Expr<float4> values) noexcept {
    auto x = sample_cie_x(lambda);
    auto y = sample_cie_y(lambda);
    auto z = sample_cie_z(lambda);
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
               average(safe_div(x * values, lambda.pdf())),
               average(safe_div(y * values, lambda.pdf())),
               average(safe_div(z * values, lambda.pdf())));
}

Float3 sampled_spectrum_to_rgb(const SampledWavelengths &lambda, Expr<float4> values) noexcept {
    return cie_xyz_to_rgb(sampled_spectrum_to_xyz(lambda, values));
}

}// namespace luisa::render
