//
// Created by Mike Smith on 2019/10/21.
//

#include <color_spaces.h>
#include <frame_data.h>
#include <ray_data.h>
#include <address_spaces.h>

using namespace metal;

inline float Mitchell1D(float x) {
    constexpr auto B = 1.0f / 3.0f;
    constexpr auto C = 1.0f / 3.0f;
    x = abs(2 * x);
    auto xx = x * x;
    return (1.0f / 6.0f) *
           (x > 1 ?
            ((-B - 6 * C) * xx + (6 * B + 30 * C) * x + (-12 * B - 48 * C)) * x + (8 * B + 24 * C) :
            ((12 - 9 * B - 6 * C) * xx + (-18 + 12 * B + 6 * C) * x) * x + (6 - 2 * B));
}

kernel void clear_frame(
    constant FrameData &frame_data [[buffer(0)]],
    texture2d<float, access::write> tex [[texture(0)]],
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        tex.write(Vec4f{}, tid);
    }
    
}

kernel void mitchell_natravali_filter(
    device const RayData *ray_buffer [[buffer(0)]],
    constant FrameData &frame_data [[buffer(1)]],
    constant uint &pixel_radius [[buffer(2)]],
    texture2d<float, access::read_write> filtered [[texture(0)]],
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        
        auto index = tid.y * frame_data.size.x + tid.x;
        auto pixel = ray_buffer[index].pixel;
        auto radiance = ray_buffer[index].radiance;
        
        auto screen = uint2(pixel);
        auto filter_radius = pixel_radius + 0.5f;
        auto inv_filter_radius = 1.0f / filter_radius;
        auto min_x = max(screen.x, pixel_radius) - pixel_radius;
        auto min_y = max(screen.y, pixel_radius) - pixel_radius;
        auto max_x = min(screen.x + pixel_radius, frame_data.size.x - 1u);
        auto max_y = min(screen.y + pixel_radius, frame_data.size.y - 1u);
        for (auto y = min_y; y <= max_y; y++) {
            for (auto x = min_x; x <= max_x; x++) {
                auto weight = Mitchell1D(abs(x + 0.5f - pixel.x) * inv_filter_radius) * Mitchell1D(abs(y + 0.5f - pixel.y) * inv_filter_radius);
                auto f = filtered.read(screen);
                filtered.write(Vec4f(Vec3f(f) + weight * radiance, f.a + weight), screen);
            }
        }
    }
    
}

kernel void accumulate(
    constant FrameData &frame_data [[buffer(0)]],
    texture2d<float, access::read> new_frame [[texture(0)]],
    texture2d<float, access::read_write> result [[texture(1)]],
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        auto f = new_frame.read(tid);
        result.write(mix(result.read(tid), Vec4f(XYZ2RGB(ACEScg2XYZ(Vec3f(Vec3f(f) / max(f.a, 1e-4f)))), 1.0f), 1.0f / (frame_data.index + 1.0f)), tid);
    }
    
}
