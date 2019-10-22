//
// Created by Mike Smith on 2019/10/21.
//

#include "../src/frame_data.h"
#include "../src/ray_data.h"

#include "../src/address_spaces.h"

using namespace metal;

kernel void accumulate(
    device RayData *ray_buffer [[buffer(0)]],
    constant FrameData &frame_data [[buffer(1)]],
    texture2d<float, access::read_write> result [[texture(0)]],
    uint2 tid [[thread_position_in_grid]]) {
    
    if (tid.x < frame_data.size.x && tid.y < frame_data.size.y) {
        auto index = tid.y * frame_data.size.x + tid.x;
        auto radiance = ray_buffer[index].radiance;
        if (frame_data.index == 0) {
            result.write(Vec4f(radiance, 1.0f), tid);
        } else {
            auto old = Vec3f(result.read(tid));
            auto t = 1.0f / (frame_data.index + 1.0f);
            result.write(Vec4f(old * (1.0f - t) + radiance * t, 1.0f), tid);
        }
    }
    
}
