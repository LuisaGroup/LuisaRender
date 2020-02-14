//
// Created by Mike Smith on 2020/2/2.
//

#include "path_tracing.h"

namespace luisa {

void PathTracing::render() {

}

PathTracing::PathTracing(Device *device, const ParameterSet &parameter_set)
    : Integrator{device, parameter_set} {}
    
}
