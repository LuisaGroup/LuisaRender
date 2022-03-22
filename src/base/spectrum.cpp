//
// Created by Mike Smith on 2022/3/21.
//

#include <util/colorspace.h>
#include <base/spectrum.h>

namespace luisa::render {

Float3 Spectrum::Instance::srgb(const SampledSpectrum &sp) const noexcept {
    auto xyz = cie_xyz(sp);
    return cie_xyz_to_linear_srgb(xyz);
}

Spectrum::Spectrum(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::SPECTRUM} {}

}// namespace luisa::render
