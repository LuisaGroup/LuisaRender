#include "compatibility.h"
#include <core/color_spaces.h>
#include <filters/mitchell_netravali_filter.h>

using namespace luisa;

// TODO: optimization
LUISA_KERNEL void mitchell_netravali_filter_add_samples(
    LUISA_DEVICE_SPACE const float3 *ray_radiance_buffer,
    LUISA_DEVICE_SPACE const float2 *ray_pixel_buffer,
    LUISA_DEVICE_SPACE const uint *ray_queue,
    LUISA_DEVICE_SPACE const uint &ray_queue_size,
    LUISA_DEVICE_SPACE Atomic<int> *accumulation_buffer,
    LUISA_UNIFORM_SPACE MitchellNetravaliFilterAddSamplesKernelUniforms &uniforms,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < ray_queue_size) {
        
        auto index = ray_queue[tid.x];
        auto pixel = ray_pixel_buffer[index];
        auto value = ACEScg2XYZ(ray_radiance_buffer[index]);
        
        auto inv_radius = 1.0f / uniforms.radius;
        
        auto x_min = static_cast<uint>(max(0.5f, floor(pixel.x - uniforms.radius)));
        auto x_max = static_cast<uint>(min(uniforms.resolution.x - 0.5f, ceil(pixel.x + uniforms.radius)));
        auto y_min = static_cast<uint>(max(0.5f, floor(pixel.y - uniforms.radius)));
        auto y_max = static_cast<uint>(min(uniforms.resolution.y - 0.5f, ceil(pixel.y + uniforms.radius)));
        
        for (auto y = y_min; y <= y_max; y++) {
            for (auto x = x_min; x <= x_max; x++) {
                auto px = x + 0.5f;
                auto py = y + 0.5f;
                auto wx = mitchell_netravali_1d((pixel.x - px) / inv_radius, uniforms.b, uniforms.c);
                auto wy = mitchell_netravali_1d((pixel.y - py) * inv_radius, uniforms.b, uniforms.c);
                auto weight = wx * wy;
                if (weight != 0.0f) {
                    auto weighted_value = make_int3(round(value * weight * 1024.0f));
                    luisa_atomic_fetch_add(accumulation_buffer[(x + y * uniforms.resolution.x) * 4u + 0u], weighted_value.x);
                    luisa_atomic_fetch_add(accumulation_buffer[(x + y * uniforms.resolution.x) * 4u + 1u], weighted_value.y);
                    luisa_atomic_fetch_add(accumulation_buffer[(x + y * uniforms.resolution.x) * 4u + 2u], weighted_value.z);
                    luisa_atomic_fetch_add(accumulation_buffer[(x + y * uniforms.resolution.x) * 4u + 3u], static_cast<int>(round(weight * 1024.0f)));
                }
            }
        }
    }
    
}
