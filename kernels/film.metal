//
// Created by Mike Smith on 2019/10/21.
//

#include "../src/frame_data.h"
#include "../src/ray_data.h"

#include "../src/address_spaces.h"

using namespace metal;

constexpr float Mitchell1D(float x) {
    constexpr auto B = 1.0f / 3.0f;
    constexpr auto C = 1.0f / 3.0f;
    x = 2.0f * (x > 0.0f ? x : -x);
    return (x > 1 ?
            (-B - 6 * C) * x * x * x + (6 * B + 30 * C) * x * x + (-12 * B - 48 * C) * x + (8 * B + 24 * C) :
            (12 - 9 * B - 6 * C) * x * x * x + (-18 + 12 * B + 6 * C) * x * x + (6 - 2 * B)) *
           (1.f / 6.f);
}

kernel void accumulate(
    device const RayData *ray_buffer [[buffer(0)]],
    constant FrameData &frame_data [[buffer(1)]],
    texture2d<float, access::read_write> result [[texture(0)]],
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        
        auto weight_sum = 0.0f;
        Vec3f radiance_sum{};
        
        auto px = tid.x + 0.5f;
        auto py = tid.y + 0.5f;
        constexpr auto pixel_radius = 2u;
        constexpr auto filter_radius = pixel_radius + 0.5f;
        constexpr auto inv_filter_radius = 1.0f / filter_radius;
        for (auto y = max(tid.y, pixel_radius) - pixel_radius; y <= min(tid.y + pixel_radius, frame_data.size.y - 1u); y++) {
            for (auto x = max(tid.x, pixel_radius) - pixel_radius; x <= min(tid.x + pixel_radius, frame_data.size.x - 1u); x++) {
                auto index = y * frame_data.size.x + x;
                auto ray_o = ray_buffer[index].pixel;
                auto weight = Mitchell1D(abs(ray_o.x - px) * inv_filter_radius) * Mitchell1D(abs(ray_o.y - py) * inv_filter_radius);
                weight_sum += weight;
                radiance_sum += weight * ray_buffer[index].radiance;
            }
        }
        auto radiance = radiance_sum / weight_sum;
        
        auto old = Vec3f(result.read(tid));
        auto t = 1.0f / (frame_data.index + 1.0f);  // should work even if index == 0
        result.write(Vec4f(old * (1.0f - t) + radiance * t, 1.0f), tid);
    }
    
}
