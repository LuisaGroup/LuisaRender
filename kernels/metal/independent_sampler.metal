#include "compatibility.h"
#include <samplers/independent.h>

LUISA_KERNEL void reset_states(
    LUISA_UNIFORM_SPACE luisa::Viewport &film_viewport,
    LUISA_DEVICE_SPACE luisa::sampler::independent::State *sampler_state_buffer,
    luisa::uint2 tid [[thread_position_in_grid]]) {
    
    luisa::sampler::independent::reset_states(film_viewport, sampler_state_buffer, tid.x);
}

LUISA_KERNEL void generate_samples(
    LUISA_DEVICE_SPACE luisa::sampler::independent::State *sampler_state_buffer,
    LUISA_DEVICE_SPACE const luisa::uint *ray_queue,
    LUISA_DEVICE_SPACE const luisa::uint &ray_count,
    LUISA_DEVICE_SPACE float *sample_buffer,
    LUISA_UNIFORM_SPACE luisa::sampler::independent::GenerateSamplesKernelUniforms &uniforms,
    luisa::uint2 tid [[thread_position_in_grid]]) {
    
    luisa::sampler::independent::generate_samples(sampler_state_buffer, ray_queue, ray_count, sample_buffer, uniforms, tid.x);
}
