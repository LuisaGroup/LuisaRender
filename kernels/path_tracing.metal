//
// Created by Mike Smith on 2019/10/21.
//

#include <random.h>
#include <frame_data.h>
#include <ray_data.h>
#include <intersection_data.h>
#include <material_data.h>
#include <light_data.h>
#include <onb.h>
#include <sampling.h>
#include <color_spaces.h>

using namespace metal;

kernel void sample_lights(
    device RayData *ray_buffer [[buffer(0)]],
    device const IntersectionData *intersection_buffer [[buffer(1)]],
    device const LightData *light_buffer [[buffer(2)]],
    device const Vec3f *p_buffer[[buffer(3)]],
    device ShadowRayData *shadow_ray_buffer [[buffer(4)]],
    constant uint &light_count [[buffer(5)]],
    device const uint &ray_count [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgsize [[threads_per_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 gsize [[threadgroups_per_grid]]) {
    
    auto index = (tgsize.x * tgsize.y) * (tgid.y * gsize.x + tgid.x) + tid;
    
    if (index < ray_count) {
        
        auto its = intersection_buffer[index];
        auto ray_seed = ray_buffer[index].seed;
        
        ShadowRayData shadow_ray{};
        
        if (ray_buffer[index].max_distance <= 0.0f || its.distance <= 0.0f) {  // no intersection
            shadow_ray.max_distance = -1.0f;  // terminate the ray
        } else {  // has an intersection
            auto uvw = Vec3f(its.barycentric, 1.0f - its.barycentric.x - its.barycentric.y);
            auto i0 = its.triangle_index * 3u;
            auto P = uvw.x * p_buffer[i0] + uvw.y * p_buffer[i0 + 1] + uvw.z * p_buffer[i0 + 2];
            auto light = light_buffer[min(static_cast<uint>(halton(ray_seed) * light_count), light_count - 1u)];
            auto L = light.position - P;
            auto dist = length(L);
            auto inv_dist = 1.0f / dist;
            shadow_ray.direction = L * inv_dist;
            shadow_ray.origin = P + 1e-4f * shadow_ray.direction;
            shadow_ray.min_distance = 0.0f;
            shadow_ray.max_distance = dist - 1e-4f;
            shadow_ray.light_radiance = XYZ2ACEScg(RGB2XYZ(light.emission)) * inv_dist * inv_dist;
            shadow_ray.light_pdf = 1.0f / light_count;
        }
        ray_buffer[index].seed = ray_seed;
        shadow_ray_buffer[index] = shadow_ray;
    }
}

kernel void trace_radiance(
    device RayData *ray_buffer [[buffer(0)]],
    device const ShadowRayData *shadow_ray_buffer [[buffer(1)]],
    device const IntersectionData *its_buffer [[buffer(2)]],
    device const ShadowIntersectionData *shadow_its_buffer [[buffer(3)]],
    device const Vec3f *p_buffer [[buffer(4)]],
    device const Vec3f *n_buffer [[buffer(5)]],
    device const uint *material_id_buffer [[buffer(6)]],
    device const MaterialData *material_buffer [[buffer(7)]],
    device const uint &ray_count [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgsize [[threads_per_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 gsize [[threadgroups_per_grid]]) {
    
    auto index = (tgsize.x * tgsize.y) * (tgid.y * gsize.x + tgid.x) + tid;
    
    if (index < ray_count) {
    
        if (ray_buffer[index].max_distance <= 0.0f || its_buffer[index].distance <= 0.0f) {  // no intersection
            ray_buffer[index].max_distance = -1.0f;  // terminate the ray
        } else {
            auto ray = ray_buffer[index];
            auto its = its_buffer[index];
            auto material = material_buffer[material_id_buffer[its.triangle_index]];
            material.albedo = XYZ2ACEScg(RGB2XYZ(material.albedo));
            auto i0 = its.triangle_index * 3;
            auto uvw = Vec3f(its.barycentric, 1.0f - its.barycentric.x - its.barycentric.y);
            auto P = uvw.x * p_buffer[i0] + uvw.y * p_buffer[i0 + 1] + uvw.z * p_buffer[i0 + 2];
            auto N = normalize(uvw.x * n_buffer[i0] + uvw.y * n_buffer[i0 + 1] + uvw.z * n_buffer[i0 + 2]);
            auto V = -ray.direction;
            
            auto NdotV = dot(N, V);
            if (NdotV < 0.0f) {
                N = -N;
                NdotV = -NdotV;
            }
            
            // direct lighting
            auto shadow_its = shadow_its_buffer[index];
            if (!material.is_mirror && shadow_its.distance < 0.0f) {  // not occluded
                auto shadow_ray = shadow_ray_buffer[index];
                auto L = shadow_ray.direction;
                auto NdotL = max(dot(N, L), 0.0f);
                ray.radiance += ray.throughput * material.albedo * NdotL * shadow_ray.light_radiance / shadow_ray.light_pdf;
            }
            
            // sampling brdf
            ray.max_distance = INFINITY;
            if (material.is_mirror) {
                ray.direction = normalize(2.0f * NdotV * N - V);
            } else {
                ray.direction = normalize(Onb{N}.inverse_transform(Vec3f(cosine_sample_hemisphere(halton(ray.seed), halton(ray.seed)))));
                ray.throughput *= material.albedo;  // simplified for lambertian materials
                ray.depth++;
                if (ray.depth > 3) {  // RR
                    auto q = max(0.05f, 1.0f - ACEScg2Luminance(ray.throughput));
                    if (halton(ray.seed) < q) {
                        ray.max_distance = -1.0f;
                    } else {
                        ray.throughput /= 1.0f - q;
                    }
                }
            }
            ray.origin = P + 1e-4f * ray.direction;
            ray.min_distance = 0.0f;
            ray_buffer[index] = ray;
        }
    }
    
}

kernel void sort_rays(
    device const RayData *ray_buffer [[buffer(0)]],
    device const uint &ray_count [[buffer(1)]],
    device RayData *output_ray_buffer [[buffer(2)]],
    device atomic_uint &output_ray_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgsize [[threads_per_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 gsize [[threadgroups_per_grid]]) {
    
    auto index = (tgsize.x * tgsize.y) * (tgid.y * gsize.x + tgid.x) + tid;
    
    if (index < ray_count) {
    
    }
}