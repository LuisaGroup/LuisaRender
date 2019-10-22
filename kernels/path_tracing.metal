//
// Created by Mike Smith on 2019/10/21.
//

#include "../src/random.h"
#include "../src/frame_data.h"
#include "../src/ray_data.h"
#include "../src/intersection_data.h"
#include "../src/material_data.h"
#include "../src/light_data.h"
#include "../src/onb.h"
#include "../src/sampling.h"
#include "../src/luminance.h"

using namespace metal;

kernel void sample_lights(
    device RayData *ray_buffer [[buffer(0)]],
    device IntersectionData *intersection_buffer [[buffer(1)]],
    device LightData *light_buffer [[buffer(2)]],
    device Vec4f *position_buffer [[buffer(3)]],
    device ShadowRayData *shadow_ray_buffer [[buffer(4)]],
    constant uint &light_count [[buffer(5)]],
    constant FrameData &frame_data [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        
        auto index = tid.y * frame_data.size.x + tid.x;
        
        auto ray = ray_buffer[index];
        auto its = intersection_buffer[index];
        
        ShadowRayData shadow_ray{};
        if (ray.max_distance <= 0.0f || its.distance <= 0.0f) {  // no intersection
            shadow_ray.max_distance = -1.0f;  // terminate the ray
        } else {  // has an intersection
            auto uvw = Vec3f(its.barycentric, 1.0f - its.barycentric.x - its.barycentric.y);
            auto i0 = its.triangle_index * 3;
            auto p = uvw.x * Vec3f(position_buffer[i0]) + uvw.y * Vec3f(position_buffer[i0 + 1]) + uvw.z * Vec3f(position_buffer[i0 + 2]);
            auto light = light_buffer[min(static_cast<uint>(halton(ray.seed) * light_count), light_count - 1u)];
            auto L = light.position - p;
            auto dist = length(L);
            auto inv_dist = 1.0f / dist;
            shadow_ray.direction = L * inv_dist;
            shadow_ray.origin = p + 1e-3f * shadow_ray.direction;
            shadow_ray.min_distance = 1e-3f;
            shadow_ray.max_distance = dist - 1e-3f;
            shadow_ray.light_radiance = light.emission * inv_dist * inv_dist;
            shadow_ray.light_pdf = 1.0f / (light_count);
        }
        ray_buffer[index].seed = ray.seed;
        shadow_ray_buffer[index] = shadow_ray;
    }
}

kernel void trace_radiance(
    device RayData *ray_buffer [[buffer(0)]],
    device ShadowRayData *shadow_ray_buffer [[buffer(1)]],
    device IntersectionData *its_buffer [[buffer(2)]],
    device ShadowIntersectionData *shadow_its_buffer [[buffer(3)]],
    device Vec3f *p_buffer [[buffer(4)]],
    device Vec3f *n_buffer [[buffer(5)]],
    device uint *material_id_buffer [[buffer(6)]],
    device MaterialData *material_buffer [[buffer(7)]],
    constant FrameData &frame_data [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        
        auto index = tid.y * frame_data.size.x + tid.x;
        auto ray = ray_buffer[index];
        auto its = its_buffer[index];
        
        if (ray.max_distance <= 0.0f || its.distance <= 0.0f) {  // no intersection
            ray.max_distance = -1.0f;  // terminate the ray
            if (ray.depth == 0) {
                ray.radiance = PackedVec3f{0.3f, 0.2f, 0.1f};  // background
            }
        } else {
            auto uvw = Vec3f(its.barycentric, 1.0f - its.barycentric.x - its.barycentric.y);
            auto i0 = its.triangle_index * 3;
            auto P = uvw.x * p_buffer[i0] + uvw.y * p_buffer[i0 + 1] + uvw.z * p_buffer[i0 + 2];
            auto material = material_buffer[material_id_buffer[its.triangle_index]];
            auto V = normalize(-ray.direction);
            auto N = normalize(uvw.x * n_buffer[i0] + uvw.y * n_buffer[i0 + 1] + uvw.z * n_buffer[i0 + 2]);
            if (dot(N, V) < 0.0f) { N = -N; }
            
            // direct lighting
            auto shadow_its = shadow_its_buffer[index];
            if (shadow_its.distance < 0.0f) {  // not occluded
                auto shadow_ray = shadow_ray_buffer[index];
                auto L = shadow_ray.direction;
                auto NdotL = max(dot(N, L), 0.0f);
                ray.radiance += ray.throughput * material.albedo * NdotL * shadow_ray.light_radiance / shadow_ray.light_pdf;
            }
            
            // sampling brdf
            auto L = normalize(Onb{N}.inverse_transform(Vec3f(cosine_sample_hemisphere(halton(ray.seed), halton(ray.seed)))));
            auto NdotL = dot(N, L);
            auto pdf = max(NdotL * M_1_PIf, 1e-3f);
            ray.direction = L;
            ray.origin = P + 1e-3f * L;
            ray.max_distance = INFINITY;
            ray.throughput *= material.albedo * NdotL * M_1_PIf / pdf;
            if (ray.depth > 3) {  // RR
                auto q = max(0.05f, 1.0f - luminance(ray.throughput));
                if (halton(ray.seed) < q) {
                    ray.max_distance = -1.0f;
                } else {
                    ray.throughput /= 1.0f - q;
                }
            }
        }
        ray.depth++;
        ray_buffer[index] = ray;
    }
    
}