#include "compatibility.h"
#include <integrators/normal_visualizer.h>

using namespace luisa;

LUISA_KERNEL void normal_visualizer_colorize_normals(
    LUISA_DEVICE_SPACE float3 *normals,
    LUISA_DEVICE_SPACE const bool *valid_buffer,
    LUISA_UNIFORM_SPACE uint &pixel_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    integrator::normal::colorize_normals(normals, valid_buffer, pixel_count, tid.x);
}
