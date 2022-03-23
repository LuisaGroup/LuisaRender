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

SampledWavelengths Spectrum::Instance::sample(Sampler::Instance &sampler) const noexcept {
    using namespace compute;
    constexpr auto sample_visible_wavelengths = [](auto u) noexcept {
        return clamp(538.0f - 138.888889f * atanh(0.85691062f - 1.82750197f * u),
                     visible_wavelength_min, visible_wavelength_max);
    };
    constexpr auto visible_wavelengths_pdf = [](auto lambda) noexcept {
        constexpr auto sqr = [](auto x) noexcept { return x * x; };
        return 0.0039398042f / sqr(cosh(0.0072f * (lambda - 538.0f)));
    };
    auto u = sampler.generate_1d();
    auto n = node()->dimension();
    SampledWavelengths swl{this};
    for (auto i = 0u; i < n; i++) {
        auto offset = static_cast<float>(i * (1.0 / n));
        auto up = fract(u + offset);
        auto lambda = sample_visible_wavelengths(up);
        swl.set_lambda(i, lambda);
        swl.set_pdf(i, visible_wavelengths_pdf(lambda));
    }
    return swl;
}

Spectrum::Spectrum(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SPECTRUM} {}

SampledWavelengths::SampledWavelengths(const Spectrum::Instance *spec) noexcept
    : _spectrum{spec},
      _lambdas{spec->node()->dimension()},
      _pdfs{spec->node()->dimension()} {}

SampledSpectrum SampledWavelengths::albedo_from_srgb(Expr<float3> rgb) const noexcept {
    return _spectrum->albedo_from_srgb(*this, rgb);
}

SampledSpectrum SampledWavelengths::illuminant_from_srgb(Expr<float3> rgb) const noexcept {
    return _spectrum->illuminant_from_srgb(*this, rgb);
}

Float SampledWavelengths::cie_y(const SampledSpectrum &s) const noexcept {
    return _spectrum->cie_y(*this, s);
}

Float3 SampledWavelengths::cie_xyz(const SampledSpectrum &s) const noexcept {
    return _spectrum->cie_xyz(*this, s);
}

Float3 SampledWavelengths::srgb(const SampledSpectrum &s) const noexcept {
    return _spectrum->srgb(*this, s);
}

Float3 SampledWavelengths::backward_albedo_from_srgb(Expr<float3> rgb, const SampledSpectrum &dSpec) const noexcept {
    return _spectrum->backward_albedo_from_srgb(*this, rgb, dSpec);
}

Float3 SampledWavelengths::backward_illuminant_from_srgb(Expr<float3> rgb, const SampledSpectrum &dSpec) const noexcept {
    return _spectrum->backward_illuminant_from_srgb(*this, rgb, dSpec);
}

SampledSpectrum SampledWavelengths::backward_cie_y(const SampledSpectrum &s, Expr<float> dY) const noexcept {
    return _spectrum->backward_cie_y(*this, s, dY);
}

SampledSpectrum SampledWavelengths::backward_cie_xyz(const SampledSpectrum &s, Expr<float3> dXYZ) const noexcept {
    return _spectrum->backward_cie_xyz(*this, s, dXYZ);
}

SampledSpectrum SampledWavelengths::backward_srgb(const SampledSpectrum &s, Expr<float3> dSRGB) const noexcept {
    return _spectrum->backward_srgb(*this, s, dSRGB);
}

}// namespace luisa::render
