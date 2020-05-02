//
// Created by Mike Smith on 2020/2/2.
//

#include <core/integrator.h>

#include "path_tracing.h"

namespace luisa {

class PathTracing : public Integrator {

protected:
    std::unique_ptr<Buffer<float2>> _ray_pixel_buffer;
    std::unique_ptr<Buffer<float3>> _ray_throughput_buffer;
    std::unique_ptr<Buffer<float3>> _ray_radiance_buffer;
    std::unique_ptr<Buffer<uint8_t>> _ray_depth_buffer;
    std::unique_ptr<Buffer<float>> _ray_pdf_buffer;
    std::unique_ptr<Buffer<Ray>> _ray_buffer;
    
    uint _max_depth;
    
    void _prepare_for_frame() override;

public:
    PathTracing(Device *device, const ParameterSet &parameter_set);
    void render_frame(KernelDispatcher &dispatch) override;
};

LUISA_REGISTER_NODE_CREATOR("PathTracing", PathTracing)

PathTracing::PathTracing(Device *device, const ParameterSet &parameter_set)
    : Integrator{device, parameter_set},
      _max_depth{parameter_set["max_depth"].parse_uint_or_default(8u)} {}

void PathTracing::render_frame(KernelDispatcher &dispatch) {

}

void PathTracing::_prepare_for_frame() {

}
    
}
