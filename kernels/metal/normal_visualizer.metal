#include "compatibility.h"
#include <integrators/normal_visualizer.h>

using namespace luisa;

LUISA_KERNEL void colorize_normals(
    LUISA_DEVICE_SPACE float3 *normals,
    LUISA_DEVICE_SPACE const float3 *throughput_buffer,
    LUISA_DEVICE_SPACE const uint8_t *state_buffer,
    LUISA_UNIFORM_SPACE uint &pixel_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    integrator::normal::colorize_normals(normals, throughput_buffer, state_buffer, pixel_count, tid.x);
}
