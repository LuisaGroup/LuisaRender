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
        return ite(b == 0.f, 0.f, a / b);
    };
    auto sum = def(make_float3());
    for (auto i = 0u; i < swl.dimension(); i++) {
        auto lambda = swl.lambda(i);
        auto pdf = swl.pdf(i);
        sum += make_float3(safe_div(_cie_x.sample(lambda) * sp[i], pdf),
                           safe_div(_cie_y.sample(lambda) * sp[i], pdf),
                           safe_div(_cie_z.sample(lambda) * sp[i], pdf));
    }
    auto denom = static_cast<float>(swl.dimension()) * SPD::cie_y_integral();
    return sum / denom;
}

SampledWavelengths Spectrum::Instance::sample(Expr<float> u) const noexcept {
    LUISA_ERROR_WITH_LOCATION("Spectrum::sample() is not implemented.");
}
//default impl: assuming |wavelength difference|<3nm is same wavelength and accumulate answer
Float3 Spectrum::Instance::wavelength_mul(const SampledWavelengths& target_swl, const SampledSpectrum& target_sp,
    const SampledWavelengths& swl, const SampledSpectrum& sp) const noexcept {
    SampledSpectrum ret_sp{target_swl.dimension()};
    SampledWavelengths ret_swl{target_swl.dimension()};
    float error_bound = 3.0f;
    for (auto i = 0u; i < target_swl.dimension(); ++i) {
        auto target_lambda = target_swl.lambda(i);
        ret_swl.set_lambda(i,target_lambda);
        auto target_pdf = target_swl.pdf(i);
        Float accum_pdf = 0.0f;
        for (auto j = 0u; j < swl.dimension(); ++j) {
            auto lambda = swl.lambda(j);
            auto pdf = swl.pdf(j);
            Bool is_same = (lambda < target_lambda + error_bound) & (lambda > target_lambda - error_bound);
            accum_pdf += ite(is_same, target_pdf * pdf,0.0f);
            ret_sp[i] += ite(is_same, target_sp[i] * sp[j],0.0f);
        }
        //The actual pdf(p(lambda)) is pdf*dimension
        ret_swl.set_pdf(i, accum_pdf*swl.dimension()*2*error_bound);
    }
    return srgb(ret_swl, ret_sp);
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
