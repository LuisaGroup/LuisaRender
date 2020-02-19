#include "compatibility.h"
#include <integrators/normal_visualizer.h>

using namespace luisa;

LUISA_KERNEL void normal_visualizer_prepare_for_frame(
    LUISA_DEVICE_SPACE uint *ray_queue,
    LUISA_UNIFORM_SPACE uint &pixel_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    integrator::normal::prepare_for_frame(ray_queue, pixel_count, tid.x);
}

LUISA_KERNEL void normal_visualizer_colorize_normals(
    LUISA_DEVICE_SPACE float3 *normals,
    LUISA_UNIFORM_SPACE uint &pixel_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    integrator::normal::colorize_normals(normals, pixel_count, tid.x);
}
