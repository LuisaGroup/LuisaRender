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
    auto sum = def(0.f);
    constexpr auto safe_div = [](auto &&a, auto &&b) noexcept {
        return ite(b == 0.0f, 0.0f, a / b);
    };
    for (auto i = 0u; i < swl.dimension(); i++) {
        sum += safe_div(_cie_y.sample(swl.lambda(i)) * sp[i], swl.pdf(i));
    }
    auto denom = static_cast<float>(swl.dimension()) * SPD::cie_y_integral();
    return sum / denom;
}

Float3 Spectrum::Instance::cie_xyz(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept {
    using namespace compute;
    constexpr auto safe_div = [](auto a, auto b) noexcept {
        return ite(b == 0.0f, 0.0f, a / b);
    };
    auto sum = def(make_float3());
    for (auto i = 0u; i < swl.dimension(); i++) {
        auto lambda = swl.lambda(i);
        auto pdf = swl.pdf(i);
        sum += make_float3(
            safe_div(_cie_x.sample(lambda) * sp[i], pdf),
            safe_div(_cie_y.sample(lambda) * sp[i], pdf),
            safe_div(_cie_z.sample(lambda) * sp[i], pdf));
    }
    auto denom = static_cast<float>(swl.dimension()) * SPD::cie_y_integral();
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

Spectrum::Instance::Instance(Pipeline &pipeline, CommandBuffer &cb,
                             const Spectrum *spec) noexcept
    : _pipeline{pipeline}, _spectrum{spec},
      _cie_x{SPD::create_cie_x(pipeline, cb)},
      _cie_y{SPD::create_cie_y(pipeline, cb)},
      _cie_z{SPD::create_cie_z(pipeline, cb)} {}

Spectrum::Spectrum(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SPECTRUM} {}

}// namespace luisa::render
