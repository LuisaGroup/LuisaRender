//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

struct PathTracingClearRayQueuesKernelUniforms {
    uint ray_queue_count;
};

struct PathTracingGeneratePixelSamplesKernelUniforms {
    uint2 frame_size;
    uint spp;
};

struct PathTracingUpdateRayStatesKernelUniforms {
    uint ray_pool_size;
};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/integrator.h>

class PathTracing {

};

#endif
