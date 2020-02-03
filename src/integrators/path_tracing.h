//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

struct PathTracingGeneratePixelSamplesKernelUniforms {
    uint2 film_resolution;
    uint samples_per_pixel;
};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/integrator.h>

namespace luisa {

class PathTracing : public Integrator {

};

}

#endif
