#include "compatibility.h"

#include <core/data_types.h>
#include <core/mathematics.h>
#include <core/sampling.h>
#include <core/ray.h>

#include <camera_data.h>
#include <frame_data.h>
#include <core/sampler.h>

using namespace luisa;

LUISA_KERNEL void pinhole_camera_generate_rays(
    LUISA_DEVICE_SPACE uint *ray_index_buffer,
    LUISA_DEVICE_SPACE Ray *ray_buffer,
    LUISA_DEVICE_SPACE uint &ray_count,
    LUISA_DEVICE_SPACE float3 *ray_throughput_buffer,
    LUISA_DEVICE_SPACE uint *ray_seed_buffer,
    LUISA_DEVICE_SPACE float3 *ray_radiance_buffer,
    LUISA_DEVICE_SPACE uint *ray_depth_buffer,
    LUISA_DEVICE_SPACE float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE float *ray_pdf_buffer,
    LUISA_UNIFORM_SPACE CameraData &camera_data,
    LUISA_UNIFORM_SPACE FrameData &frame_data,
    uint2 tid [[thread_position_in_grid]]) {
    
    auto w = frame_data.size.x;
    auto h = frame_data.size.y;
    
    if (tid.x == 0u && tid.y == 0u) {
        ray_count = w * h;
    }
    
    if (tid.x < w && tid.y < h) {
        
        auto ray_index = tid.y * w + tid.x;
        ray_index_buffer[ray_index] = ray_index;
        
        auto seed = (tea<5>(tid.x, tid.y) + frame_data.index) << 8u;
        auto half_sensor_height = tan(0.5f * camera_data.fov) * camera_data.focal_distance;
        auto half_sensor_width = half_sensor_height * w / h;
        
        auto px = static_cast<float>(tid.x) + halton(seed);
        auto py = static_cast<float>(tid.y) + halton(seed);
        
        auto focal_plane_p = make_float3((1.0f - px / w * 2.0f) * half_sensor_width, (1.0f - py / h * 2.0f) * half_sensor_height, camera_data.focal_distance);
        auto origin = make_float3(camera_data.aperture / camera_data.near_plane * concentric_sample_disk(halton(seed), halton(seed)), 0.0f);
        auto d = focal_plane_p - origin;
        
        Ray ray{};
        ray.origin = origin.x * camera_data.left + origin.y * camera_data.up + origin.z * camera_data.front + camera_data.position;
        ray.direction = normalize(d.x * camera_data.left + d.y * camera_data.up + d.z * camera_data.front);
        ray.min_distance = 0.0f;
        ray.max_distance = INFINITY;
        
        ray_buffer[ray_index] = ray;
        ray_seed_buffer[ray_index] = seed;
        ray_pixel_buffer[ray_index] = make_float2(px, py);
        ray_throughput_buffer[ray_index] = make_float3(1.0f);
        ray_radiance_buffer[ray_index] = {};
        ray_depth_buffer[ray_index] = 0;
        ray_pdf_buffer[ray_index] = 1.0f;
    }
}
