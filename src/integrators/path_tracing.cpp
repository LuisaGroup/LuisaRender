//
// Created by Mike Smith on 2020/2/2.
//

#include "path_tracing.h"

namespace luisa {

PathTracing::PathTracing(Device *device, const ParameterSet &parameter_set)
    : Integrator{device, parameter_set} {}

void PathTracing::render_frame(Viewport viewport, Scene &scene, Camera &camera, Sampler &sampler) {

}
    
}
