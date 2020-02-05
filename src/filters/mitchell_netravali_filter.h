//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <core/data_types.h>
#include <core/mathematics.h>

namespace luisa {

struct MitchellNetravaliFilterAddSamplesKernelUniforms {
    uint2 resolution;
    float radius;
    float b;
    float c;
};

LUISA_DEVICE_CALLABLE inline float mitchell_netravali_1d(float x, float b, float c) noexcept {
    x = min(abs(2 * x), 2.0f);
    auto xx = x * x;
    return (1.0f / 6.0f) *
           (x > 1 ?
            ((-b - 6 * c) * xx + (6 * b + 30 * c) * x + (-12 * b - 48 * c)) * x + (8 * b + 24 * c) :
            ((12 - 9 * b - 6 * c) * xx + (-18 + 12 * b + 6 * c) * x) * x + (6 - 2 * b));
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/filter.h>
#include <core/film.h>

namespace luisa {

class MitchellNetravaliFilter : public Filter {

protected:
    float _b;
    float _c;
    std::unique_ptr<Kernel> _add_samples_kernel;

public:
    MitchellNetravaliFilter(Device *device, const ParameterSet &parameters)
        : Filter{device, parameters},
          _b{parameters["b"].parse_float_or_default(1.0f / 3.0f)},
          _c{parameters["c"].parse_float_or_default(1.0f / 3.0f)} {
        
        _add_samples_kernel = device->create_kernel("mitchell_netravali_filter_add_samples");
    }
    void add_samples(KernelDispatcher &dispatch, RayPool &ray_pool, RayQueueView ray_queue, Film &film) override;
    
};

LUISA_REGISTER_NODE_CREATOR("MitchellNetravali", MitchellNetravaliFilter)

}

#endif
