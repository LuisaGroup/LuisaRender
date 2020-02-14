//
// Created by Mike Smith on 2019/10/21.
//

#include "compatibility.h"
#include <integrators/path_tracing.h>

using namespace luisa;

LUISA_KERNEL void path_tracing_clear_ray_queues(
    LUISA_DEVICE_SPACE uint *queue_sizes,
    LUISA_UNIFORM_SPACE uint &ray_queue_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < ray_queue_count) {
        queue_sizes[tid.x] = 0u;
    }
}

LUISA_KERNEL void path_tracing_generate_pixel_samples(
    LUISA_DEVICE_SPACE RayState *ray_state_buffer,
    LUISA_DEVICE_SPACE SamplerState *ray_sampler_state_buffer,
    LUISA_DEVICE_SPACE uint8_t *ray_depth_buffer,
    LUISA_DEVICE_SPACE float3 *ray_radiance_buffer,
    LUISA_DEVICE_SPACE uint2 &global_pixel_sample_count,
    LUISA_DEVICE_SPACE float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE const uint *ray_queue,
    LUISA_DEVICE_SPACE const uint &ray_queue_size,
    LUISA_UNIFORM_SPACE path_tracing::GeneratePixelSamplesKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < ray_queue_size) {
        
        auto global_sample_index = make_u64(global_pixel_sample_count) - ray_queue_size + tid.x;
        auto pixel_sample_index = static_cast<uint>(global_sample_index % uniforms.samples_per_pixel);
        auto pixel_id = static_cast<uint>(global_sample_index / uniforms.samples_per_pixel);
        auto pixel_x = pixel_id % uniforms.film_resolution.x;
        auto pixel_y = pixel_id / uniforms.film_resolution.y;
        
        auto ray_index = ray_queue[tid.x];
        if (pixel_y < uniforms.film_resolution.y) {
            auto sampler_state = (tea<5>(pixel_x, pixel_y) + pixel_sample_index) << 8u;
            auto px = pixel_x + sampler_generate_sample(sampler_state);
            auto py = pixel_y + sampler_generate_sample(sampler_state);
            ray_pixel_buffer[ray_index] = make_float2(px, py);
            ray_sampler_state_buffer[ray_index] = sampler_state;
            ray_depth_buffer[ray_index] = 0u;
            ray_radiance_buffer[ray_index] = make_float3();
            ray_state_buffer[ray_index] = RayState::GENERATED;
        } else {
            ray_state_buffer[ray_index] = RayState::INVALIDATED;
        }
    }
}

LUISA_KERNEL void path_tracing_russian_roulette(
    LUISA_DEVICE_SPACE RayState *ray_state_buffer,
    LUISA_DEVICE_SPACE float3 *ray_throughput_buffer,
    LUISA_DEVICE_SPACE uint8_t *ray_depth_buffer,
    LUISA_DEVICE_SPACE SamplerState *ray_sampler_state_buffer,
    LUISA_DEVICE_SPACE const uint *ray_queue,
    LUISA_DEVICE_SPACE const uint &ray_queue_size,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < ray_queue_size) {
        
        auto ray_index = ray_queue[tid.x];
        
        auto depth = ray_depth_buffer[ray_index];
        auto throughput = ray_throughput_buffer[ray_index];
        auto sampler_state = ray_sampler_state_buffer[ray_index];
        auto q = max_component(throughput);
        auto p = max(0.05f, q);
        
        if (q <= 0.0f || (depth > 3u && p < sampler_generate_sample(sampler_state))) {  // terminate
            ray_state_buffer[ray_index] = RayState::FINISHED;
        } else {
            ray_state_buffer[ray_index] = RayState::EXTENDED;
            ray_depth_buffer[ray_index] = depth + 1u;
            ray_throughput_buffer[ray_index] = throughput * (1.0f / p);
        }
        ray_sampler_state_buffer[ray_index] = sampler_state;
    }

}

#define queue_emplace(queue, queue_size, element)  static_cast<void>(queue[luisa_atomic_fetch_add(queue_size, 1u)] = element)

LUISA_KERNEL void path_tracing_update_ray_states(
    LUISA_DEVICE_SPACE RayState *ray_state_buffer,
    LUISA_DEVICE_SPACE uint *camera_queue,
    LUISA_DEVICE_SPACE Atomic<uint> &camera_queue_size,
    LUISA_DEVICE_SPACE uint *tracing_queue,
    LUISA_DEVICE_SPACE Atomic<uint> &tracing_queue_size,
    LUISA_DEVICE_SPACE uint *gathering_queue,
    LUISA_DEVICE_SPACE Atomic<uint> &gathering_queue_size,
    LUISA_DEVICE_SPACE uint *shading_queues,
    LUISA_DEVICE_SPACE Atomic<uint> &shading_queue_size,
    LUISA_DEVICE_SPACE AtomicCounter &global_pixel_sample_count,
    LUISA_UNIFORM_SPACE uint &ray_pool_size,
    uint2 tid [[thread_position_in_grid]]) {
    
    auto index = tid.x;
    if (index < ray_pool_size) {
        switch (ray_state_buffer[index]) {
            case RayState::UNINITIALIZED: {
                luisa_atomic_counter_increase(global_pixel_sample_count);
                queue_emplace(camera_queue, camera_queue_size, index);
                // NOTE: Ray state will be updated in the path_tracing_generate_pixel_samples kernel.
                break;
            }
            case RayState::GENERATED: {
                queue_emplace(tracing_queue, tracing_queue_size, index);
                ray_state_buffer[index] = RayState::TRACED;
                break;
            }
            case RayState::TRACED: {
                queue_emplace(shading_queues, shading_queue_size, index);
                // NOTE: Ray state will be updated in the path_tracing_russian_roulette kernel
                break;
            }
            case RayState::EXTENDED: {
                queue_emplace(tracing_queue, tracing_queue_size, index);
                ray_state_buffer[index] = RayState::GENERATED;
                break;
            }
            case RayState::FINISHED: {
                queue_emplace(gathering_queue, gathering_queue_size, index);
                ray_state_buffer[index] = RayState::UNINITIALIZED;
                break;
            }
            case RayState::INVALIDATED: {
                // no more rays, just idle
                break;
            }
        }
    }
}
