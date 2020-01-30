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
    x = min(abs(2 * x), 2.0f);
    auto xx = x * x;
    return (1.0f / 6.0f) *
           (x > 1 ?
            ((-B - 6 * C) * xx + (6 * B + 30 * C) * x + (-12 * B - 48 * C)) * x + (8 * B + 24 * C) :
            ((12 - 9 * B - 6 * C) * xx + (-18 + 12 * B + 6 * C) * x) * x + (6 - 2 * B));
}

kernel void rgb_film_clear(
    device uint4 *accum_buffer,
    constant uint &ray_count,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < ray_count) {
        accum_buffer[tid.x] = {};
    }
    
}

kernel void rgb_film_gather_rays(
    device const GatherRayData *ray_buffer,
    constant FrameData &frame_data,
    device atomic_int *accum_buffer,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        auto index = tid.x + tid.y * frame_data.size.x;
        auto new_value = int4(int3(round(ACEScg2XYZ(ray_buffer[index].radiance) * 1024.0f)), 1);
        atomic_fetch_add_explicit(&accum_buffer[index * 4u + 0u], new_value.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&accum_buffer[index * 4u + 1u], new_value.y, memory_order_relaxed);
        atomic_fetch_add_explicit(&accum_buffer[index * 4u + 2u], new_value.z, memory_order_relaxed);
        atomic_fetch_add_explicit(&accum_buffer[index * 4u + 3u], new_value.w, memory_order_relaxed);
    }
}

kernel void rgb_film_convert_colorspace(
    constant FrameData &frame_data,
    device const int4 *accum_buffer,
    texture2d<float, access::write> result,
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        auto index = tid.x + tid.y * frame_data.size.x;
        auto f = Vec4f(accum_buffer[index]);
        result.write(Vec4f(XYZ2RGB(Vec3f(f) / (1024.0f * f.a)), 1.0f), tid);
    }
    
}

kernel void mitchell_natravali_filter(
    device const GatherRayData *ray_buffer [[buffer(0)]],
    constant FrameData &frame_data [[buffer(1)]],
    constant uint &pixel_radius [[buffer(2)]],
    texture2d<float, access::read_write> result [[texture(0)]],
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        
        auto index = tid.x + tid.y * frame_data.size.x;
        auto new_value = int4(int3(round(ray_buffer[index].radiance * 1024.0f)), 1);
        auto old_value = as_type<int4>(result.read(tid));
        result.write(as_type<Vec4f>(new_value + old_value), tid);
        
//        auto min_x = max(tid.x, pixel_radius) - pixel_radius;
//        auto min_y = max(tid.y, pixel_radius) - pixel_radius;
//        auto max_x = min(tid.x + pixel_radius, frame_data.size.x - 1u);
//        auto max_y = min(tid.y + pixel_radius, frame_data.size.y - 1u);
//
//        auto filter_radius = pixel_radius + 0.5f;
//        auto inv_filter_radius = 1.0f / filter_radius;
//
//        auto radiance_sum = Vec3f(0.0f);
//        auto weight_sum = 0.0f;
//        auto center = Vec2f(tid) + 0.5f;
//        for (auto y = min_y; y <= max_y; y++) {
//            for (auto x = min_x; x <= max_x; x++) {
//                auto index = y * frame_data.size.x + x;
//                auto radiance = ray_buffer[index].radiance;
//                auto pixel = ray_buffer[index].pixel;
//                auto dx = (center.x - pixel.x) * inv_filter_radius;
//                auto dy = (center.y - pixel.y) * inv_filter_radius;
//                auto weight = Mitchell1D(dx) * Mitchell1D(dy);
//                radiance_sum += weight * radiance;
//                weight_sum += weight;
//            }
//        }
//        result.write(mix(result.read(tid), Vec4f(radiance_sum, weight_sum), 1.0f / (frame_data.index + 1.0f)), tid);
    }
}

kernel void convert_colorspace_rgb(
    constant FrameData &frame_data [[buffer(0)]],
    texture2d<float, access::read_write> result [[texture(0)]],
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        auto f = Vec4f(as_type<int4>(result.read(tid)));
//        if (f.a == 0.0f) { f.a = 1e-3f; }
        result.write(Vec4f(XYZ2RGB(ACEScg2XYZ(Vec3f(f) / (1024.0f * f.a))), 1.0f), tid);
    }
    
}
