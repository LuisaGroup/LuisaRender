//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/data_types.h>

namespace luisa {

struct MitchellNetravaliFilterAddSamplesKernelUniforms {
    uint2 film_resolution;
    float a;
    float b;
    float c;
};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/filter.h>
#include <core/film.h>

namespace luisa {

class MitchellNetravaliFilter : public Filter {

protected:
    float _a;
    float _b;
    float _c;
    std::unique_ptr<Kernel> _add_samples_kernel;

public:
    void add_samples(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue, Film &film) override;
    
};

}

#endif
