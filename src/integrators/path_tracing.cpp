//
// Created by Mike Smith on 2020/2/2.
//

#include "path_tracing.h"

namespace luisa {

LUISA_REGISTER_NODE_CREATOR("PathTracing", PathTracing)

PathTracing::PathTracing(Device *device, const ParameterSet &parameter_set)
    : Integrator{device, parameter_set},
      _max_depth{parameter_set["max_depth"].parse_uint_or_default(8u)} {}

void PathTracing::render_frame(KernelDispatcher &dispatch) {

}

void PathTracing::_prepare_for_frame() {

}
    
}
