//
// Created by Mike Smith on 2020/2/14.
//

#pragma once

#include <core/viewport.h>
#include <compute/mathematics.h>

namespace luisa::sampler::independent {

using State = uint;

struct GenerateSamplesKernelUniforms {
    Viewport tile_viewport;
    Viewport film_viewport;
    uint num_dimensions;
    bool uses_ray_queue;
};

}
