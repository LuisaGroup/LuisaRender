//
// Created by Mike on 2021/12/8.
//

#include <base/sampler.h>

namespace luisa::render {

Sampler &Sampler::set_resolution(uint2 r) noexcept {
    _resolution = r;
    _on_set_resolution();
    return *this;
}

Sampler &Sampler::set_sample_count(uint spp) noexcept {
    _sample_count = spp;
    _on_set_sample_count();
    return *this;
}

}
