#include "compatibility.h"
#include <samplers/independent_sampler.h>

LUISA_KERNEL void independent_sampler_reset_states(
    LUISA_UNIFORM_SPACE luisa::Viewport &film_viewport,
    LUISA_DEVICE_SPACE luisa::sampler::independent::State *sampler_state_buffer,
    luisa::uint2 tid [[thread_position_in_grid]]) {
    
    luisa::sampler::independent::reset_states(film_viewport, sampler_state_buffer, tid.x);
}

LUISA_KERNEL void independent_sampler_generate_1d_samples(
    LUISA_DEVICE_SPACE luisa::sampler::independent::State *sampler_state_buffer,
    LUISA_DEVICE_SPACE const uint *ray_queue,
    LUISA_DEVICE_SPACE const uint &ray_count,
    LUISA_DEVICE_SPACE float *sample_buffer,
    LUISA_UNIFORM_SPACE luisa::sampler::independent::GenerateSamplesKernelUniforms &uniforms,
    luisa::uint2 tid [[thread_position_in_grid]]) {
    
    luisa::sampler::independent::generate_samples<1u>(sampler_state_buffer, ray_queue, ray_count, sample_buffer, uniforms, tid.x);
}

LUISA_KERNEL void independent_sampler_generate_2d_samples(
    LUISA_DEVICE_SPACE luisa::sampler::independent::State *sampler_state_buffer,
    LUISA_DEVICE_SPACE const uint *ray_queue,
    LUISA_DEVICE_SPACE const uint &ray_count,
    LUISA_DEVICE_SPACE float *sample_buffer,
    LUISA_UNIFORM_SPACE luisa::sampler::independent::GenerateSamplesKernelUniforms &uniforms,
    luisa::uint2 tid [[thread_position_in_grid]]) {
    
    luisa::sampler::independent::generate_samples<2u>(sampler_state_buffer, ray_queue, ray_count, sample_buffer, uniforms, tid.x);
}

LUISA_KERNEL void independent_sampler_generate_3d_samples(
    LUISA_DEVICE_SPACE luisa::sampler::independent::State *sampler_state_buffer,
    LUISA_DEVICE_SPACE const uint *ray_queue,
    LUISA_DEVICE_SPACE const uint &ray_count,
    LUISA_DEVICE_SPACE float *sample_buffer,
    LUISA_UNIFORM_SPACE luisa::sampler::independent::GenerateSamplesKernelUniforms &uniforms,
    luisa::uint2 tid [[thread_position_in_grid]]) {
    
    luisa::sampler::independent::generate_samples<3u>(sampler_state_buffer, ray_queue, ray_count, sample_buffer, uniforms, tid.x);
}

LUISA_KERNEL void independent_sampler_generate_4d_samples(
    LUISA_DEVICE_SPACE luisa::sampler::independent::State *sampler_state_buffer,
    LUISA_DEVICE_SPACE const uint *ray_queue,
    LUISA_DEVICE_SPACE const uint &ray_count,
    LUISA_DEVICE_SPACE float *sample_buffer,
    LUISA_UNIFORM_SPACE luisa::sampler::independent::GenerateSamplesKernelUniforms &uniforms,
    luisa::uint2 tid [[thread_position_in_grid]]) {
    
    luisa::sampler::independent::generate_samples<4u>(sampler_state_buffer, ray_queue, ray_count, sample_buffer, uniforms, tid.x);
}

LUISA_KERNEL void independent_sampler_generate_camera_samples(
    LUISA_DEVICE_SPACE luisa::sampler::independent::State *sampler_state_buffer,
    LUISA_DEVICE_SPACE luisa::float4 *sample_buffer,
    LUISA_UNIFORM_SPACE luisa::sampler::independent::GenerateSamplesKernelUniforms &uniforms,
    luisa::uint2 tid [[thread_position_in_grid]]) {
    
    luisa::sampler::independent::generate_camera_samples(sampler_state_buffer, sample_buffer, uniforms, tid.x);
}
