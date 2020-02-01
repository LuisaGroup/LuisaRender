//
// Created by Mike Smith on 2019/10/21.
//

#include "compatibility.h"

#include <core/data_types.h>
#include <core/ray.h>

#include <random.h>
#include <intersection_data.h>
#include <material_data.h>
#include <light_data.h>
#include <onb.h>
#include <sampling.h>
#include <color_spaces.h>

using namespace luisa;
using namespace luisa::math;

#define queue_emplace(queue, queue_size, element)  static_cast<void>(queue[luisa_atomic_fetch_add(queue_size, 1u)] = element)

LUISA_KERNEL void path_tracing_clear_ray_queue_sizes(
    LUISA_DEVICE_SPACE uint *queue_sizes,
    LUISA_PRIVATE_SPACE uint &ray_queue_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < ray_queue_count) {
        queue_sizes[tid.x] = 0u;
    }
}

LUISA_KERNEL void path_tracing_update_ray_states(
    LUISA_DEVICE_SPACE RayState *ray_state_buffer,
    LUISA_DEVICE_SPACE const bool *ray_done_buffer,
    LUISA_DEVICE_SPACE uint *camera_queue,
    LUISA_DEVICE_SPACE Atomic<uint> &camera_queue_size,
    LUISA_DEVICE_SPACE uint *tracing_queue,
    LUISA_DEVICE_SPACE Atomic<uint> &tracing_queue_size,
    LUISA_DEVICE_SPACE uint *gather_queue,
    LUISA_DEVICE_SPACE Atomic<uint> &gather_queue_size,
    LUISA_DEVICE_SPACE uint *shading_queues,
    LUISA_DEVICE_SPACE Atomic<uint> &shading_queue_size,
    LUISA_DEVICE_SPACE AtomicCounter &global_pixel_sample_count,
    LUISA_PRIVATE_SPACE uint &ray_pool_size,
    uint2 tid [[thread_position_in_grid]]) {
    
    auto index = tid.x;
    if (index < ray_pool_size) {
        switch (ray_state_buffer[index]) {
            case RayState::UNINITIALIZED: {
                luisa_atomic_counter_increase(global_pixel_sample_count);
                queue_emplace(camera_queue, camera_queue_size, index);
                ray_state_buffer[index] = RayState::GENERATED;
                break;
            }
            case RayState::GENERATED: {
                if (ray_done_buffer[index]) {  // no more rays can be generated... TODO: smarter criteria
                    ray_state_buffer[index] = RayState::INVALIDATED;
                } else {
                    queue_emplace(tracing_queue, tracing_queue_size, index);
                    ray_state_buffer[index] = RayState::TRACED;
                }
                break;
            }
            case RayState::TRACED: {
                queue_emplace(shading_queues, shading_queue_size, index);
                ray_state_buffer[index] = RayState::SHADED;
                break;
            }
            case RayState::SHADED: {
                if (ray_done_buffer[index]) {
                    queue_emplace(gather_queue, gather_queue_size, index);
                    ray_state_buffer[index] = RayState::UNINITIALIZED;
                } else {
                    queue_emplace(tracing_queue, tracing_queue_size, index);
                    ray_state_buffer[index] = RayState::TRACED;
                }
            }
            case RayState::INVALIDATED:
                break;
        }
    }
}

LUISA_KERNEL void sample_lights(
    LUISA_DEVICE_SPACE const uint *ray_index_buffer,
    LUISA_DEVICE_SPACE const Ray *ray_buffer,
    LUISA_DEVICE_SPACE uint *ray_seed_buffer,
    LUISA_DEVICE_SPACE const IntersectionData *intersection_buffer,
    LUISA_DEVICE_SPACE const LightData *light_buffer,
    LUISA_DEVICE_SPACE const float3 *p_buffer,
    LUISA_DEVICE_SPACE Ray *shadow_ray_buffer,
    LUISA_DEVICE_SPACE LightSample *light_sample_buffer,
    LUISA_PRIVATE_SPACE uint &light_count,
    LUISA_DEVICE_SPACE const uint &ray_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    auto thread_index = tid.x;
    
    if (thread_index < ray_count) {
        
        auto ray_index = ray_index_buffer[thread_index];
        auto its = intersection_buffer[thread_index];
        
        if (ray_buffer[ray_index].max_distance <= 0.0f || its.distance <= 0.0f) {  // no intersection
            shadow_ray_buffer[ray_index].max_distance = -1.0f;
        } else {  // has an intersection
            auto uvw = float3(its.barycentric, 1.0f - its.barycentric.x - its.barycentric.y);
            auto i0 = its.triangle_index * 3u;
            auto P = uvw.x * p_buffer[i0] + uvw.y * p_buffer[i0 + 1] + uvw.z * p_buffer[i0 + 2];
            auto seed = ray_seed_buffer[ray_index];
            auto light = light_buffer[min(static_cast<uint>(halton(seed) * light_count), light_count - 1u)];
            ray_seed_buffer[ray_index] = seed;
            auto L = light.position - P;
            auto dist = length(L);
            auto inv_dist = 1.0f / dist;
            auto wi = normalize(L);
            
            Ray shadow_ray{};
            shadow_ray.direction = wi;
            shadow_ray.origin = P + 1e-4f * wi;
            shadow_ray.min_distance = 0.0f;
            shadow_ray.max_distance = dist - 1e-4f;
            shadow_ray_buffer[thread_index] = shadow_ray;
            
            LightSample light_sample{};
            light_sample.radiance = XYZ2ACEScg(RGB2XYZ(light.emission)) * inv_dist * inv_dist * static_cast<float>(light_count);
            light_sample.pdf = 1.0f;
            light_sample.direction = wi;
            light_sample.distance = dist;
            light_sample_buffer[thread_index] = light_sample;
        }
    }
}

LUISA_KERNEL void trace_radiance(
    LUISA_DEVICE_SPACE const uint *ray_index_buffer,
    LUISA_DEVICE_SPACE Ray *ray_buffer,
    LUISA_DEVICE_SPACE float3 *ray_radiance_buffer,
    LUISA_DEVICE_SPACE float3 *ray_throughput_buffer,
    LUISA_DEVICE_SPACE uint *ray_seed_buffer,
    LUISA_DEVICE_SPACE uint *ray_depth_buffer,
    LUISA_DEVICE_SPACE const LightSample *light_sample_buffer,
    LUISA_DEVICE_SPACE const IntersectionData *its_buffer,
    LUISA_DEVICE_SPACE const ShadowIntersectionData *shadow_its_buffer,
    LUISA_DEVICE_SPACE const float3 *p_buffer,
    LUISA_DEVICE_SPACE const float3 *n_buffer,
    LUISA_DEVICE_SPACE const uint *material_id_buffer,
    LUISA_DEVICE_SPACE const MaterialData *material_buffer,
    LUISA_DEVICE_SPACE const uint &ray_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    auto thread_index = tid.x;
    if (thread_index < ray_count) {
        
        auto ray_index = ray_index_buffer[thread_index];
        auto ray = ray_buffer[ray_index];
        if (ray.max_distance <= 0.0f || its_buffer[thread_index].distance <= 0.0f) {  // no intersection
            ray_buffer[ray_index].max_distance = -1.0f;  // terminate the ray
        } else {
            auto its = its_buffer[thread_index];
            auto material = material_buffer[material_id_buffer[its.triangle_index]];
            auto albedo = XYZ2ACEScg(RGB2XYZ(make_float3(material.albedo)));
            auto i0 = its.triangle_index * 3;
            auto uvw = float3(its.barycentric, 1.0f - its.barycentric.x - its.barycentric.y);
            auto P = uvw.x * p_buffer[i0] + uvw.y * p_buffer[i0 + 1] + uvw.z * p_buffer[i0 + 2];
            auto N = normalize(uvw.x * n_buffer[i0] + uvw.y * n_buffer[i0 + 1] + uvw.z * n_buffer[i0 + 2]);
            auto V = -make_float3(ray.direction);
            
            auto NdotV = dot(N, V);
            if (NdotV < 0.0f) {
                N = -N;
                NdotV = -NdotV;
            }
            
            // direct lighting
            auto shadow_its = shadow_its_buffer[thread_index];
            auto ray_throughput = ray_throughput_buffer[ray_index];
            if (!material.is_mirror && shadow_its.distance < 0.0f) {  // not occluded
                auto light_sample = light_sample_buffer[thread_index];
                auto NdotL = max(dot(N, make_float3(light_sample.direction)), 0.0f);
                ray_radiance_buffer[ray_index] += ray_throughput * albedo * NdotL * make_float3(light_sample.radiance) / light_sample.pdf;
            }
            
            // sampling brdf
            ray.max_distance = INFINITY;
            if (material.is_mirror) {
                ray.direction = normalize(2.0f * NdotV * N - V);
            } else {
                auto seed = ray_seed_buffer[ray_index];
                ray.direction = normalize(Onb{N}.inverse_transform(float3(cosine_sample_hemisphere(halton(seed), halton(seed)))));
                ray_throughput *= albedo;  // simplified for lambertian materials
                auto ray_depth = (++ray_depth_buffer[ray_index]);
                if (ray_depth > 3) {  // RR
                    auto q = max(0.05f, 1.0f - max_component(ray_throughput));
                    if (halton(seed) < q) {
                        ray.max_distance = -1.0f;
                    } else {
                        ray_throughput /= 1.0f - q;
                    }
                }
                ray_seed_buffer[ray_index] = seed;
            }
            ray_throughput_buffer[ray_index] = ray_throughput;
            ray.origin = P + make_float3(1e-4f * ray.direction);
            ray.min_distance = 0.0f;
            ray_buffer[ray_index] = ray;
        }
    }
}

LUISA_KERNEL void sort_rays(
    LUISA_DEVICE_SPACE const uint *ray_index_buffer,
    LUISA_DEVICE_SPACE const Ray *ray_buffer,
    LUISA_DEVICE_SPACE uint *output_ray_index_buffer,
    LUISA_DEVICE_SPACE const uint &ray_count,
    LUISA_DEVICE_SPACE Atomic<uint> &output_ray_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    auto thread_index = tid.x;
    if (thread_index < ray_count) {
        auto ray_index = ray_index_buffer[thread_index];
        if (ray_buffer[ray_index].max_distance > 0.0f) {  // add active rays to next bounce
            auto output_index = luisa_atomic_fetch_add(output_ray_count, 1u);
            output_ray_index_buffer[output_index] = ray_index;
        }
    }
}
