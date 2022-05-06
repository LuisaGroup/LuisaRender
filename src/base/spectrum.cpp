//
// Created by Mike Smith on 2022/3/21.
//

#include <util/colorspace.h>
#include <base/spectrum.h>
#include <util/spec.h>

namespace luisa::render {

Float3 Spectrum::Instance::srgb(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept {
    auto xyz = cie_xyz(swl, sp);
    return cie_xyz_to_linear_srgb(xyz);
}

Float Spectrum::Instance::cie_y(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept {
    using namespace compute;
    auto &&y = DenselySampledSpectrum::cie_y();
    auto sum = def(0.f);
    constexpr auto safe_div = [](auto &&a, auto &&b) noexcept {
        return ite(b == 0.0f, 0.0f, a / b);
    };
    for (auto i = 0u; i < swl.dimension(); i++) {
        sum += safe_div(y.sample(swl.lambda(i)) * sp[i], swl.pdf(i));
    }
    auto denom = static_cast<float>(swl.dimension()) *
                 DenselySampledSpectrum::cie_y_integral();
    return sum / denom;
}

Float3 Spectrum::Instance::cie_xyz(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept {
    using namespace compute;
    auto &&x = DenselySampledSpectrum::cie_x();
    auto &&y = DenselySampledSpectrum::cie_y();
    auto &&z = DenselySampledSpectrum::cie_z();
    constexpr auto safe_div = [](auto a, auto b) noexcept {
        return ite(b == 0.0f, 0.0f, a / b);
    };
    auto sum = def(make_float3());
    for (auto i = 0u; i < swl.dimension(); i++) {
        auto lambda = swl.lambda(i);
        auto pdf = swl.pdf(i);
        sum += make_float3(
            safe_div(x.sample(lambda) * sp[i], pdf),
            safe_div(y.sample(lambda) * sp[i], pdf),
            safe_div(z.sample(lambda) * sp[i], pdf));
    }
    auto denom = static_cast<float>(swl.dimension()) *
                 DenselySampledSpectrum::cie_y_integral();
    return sum / denom;
}

SampledWavelengths Spectrum::Instance::sample(Expr<float> u) const noexcept {
    LUISA_ASSERT(!node()->is_fixed(), "Fixed spectra should not sample.");
    using namespace compute;
    constexpr auto sample_visible_wavelengths = [](auto u) noexcept {
        return clamp(538.0f - 138.888889f * atanh(0.85691062f - 1.82750197f * u),
                     visible_wavelength_min, visible_wavelength_max);
    };
    constexpr auto visible_wavelengths_pdf = [](auto lambda) noexcept {
        constexpr auto sqr = [](auto x) noexcept { return x * x; };
        return 0.0039398042f / sqr(cosh(0.0072f * (lambda - 538.0f)));
    };
    auto n = node()->dimension();
    SampledWavelengths swl{node()->dimension()};
    for (auto i = 0u; i < n; i++) {
        auto offset = static_cast<float>(i * (1.0 / n));
        auto up = fract(u + offset);
        auto lambda = sample_visible_wavelengths(up);
        swl.set_lambda(i, lambda);
        swl.set_pdf(i, visible_wavelengths_pdf(lambda));
    }
    return swl;
}

void Spectrum::Instance::_report_backward_unsupported_or_not_implemented() const noexcept {
    if (node()->is_differentiable()) {
        LUISA_ERROR_WITH_LOCATION("Backward propagation is not implemented.");
    } else {
        LUISA_ERROR_WITH_LOCATION(
            "Backward propagation is not supported "
            "in the '{}' Spectrum implementation.",
            node()->impl_type());
    }
}

Float4 Spectrum::Instance::backward_decode_albedo(
    const SampledWavelengths &swl, Expr<float4> v, const SampledSpectrum &dSpec) const noexcept {
    _report_backward_unsupported_or_not_implemented();
}

Float4 Spectrum::Instance::backward_decode_illuminant(
    const SampledWavelengths &swl, Expr<float4> v, const SampledSpectrum &dSpec) const noexcept {
    _report_backward_unsupported_or_not_implemented();
}

SampledSpectrum Spectrum::Instance::backward_cie_y(
    const SampledWavelengths &swl, const SampledSpectrum &sp, Expr<float> dY) const noexcept {
    _report_backward_unsupported_or_not_implemented();
}

SampledSpectrum Spectrum::Instance::backward_cie_xyz(
    const SampledWavelengths &swl, const SampledSpectrum &sp, Expr<float3> dXYZ) const noexcept {
    _report_backward_unsupported_or_not_implemented();
}

SampledSpectrum Spectrum::Instance::backward_srgb(
    const SampledWavelengths &swl, const SampledSpectrum &sp, Expr<float3> dSRGB) const noexcept {
    _report_backward_unsupported_or_not_implemented();
}

Spectrum::Instance::Instance(const Pipeline &pipeline, const Spectrum *spec) noexcept
    : _pipeline{pipeline}, _spectrum{spec} {}

Spectrum::Spectrum(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SPECTRUM} {}

}// namespace luisa::render
