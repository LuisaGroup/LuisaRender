#include "../src/ray_data.h"
#include "../src/camera_data.h"
#include "../src/frame_data.h"
#include "../src/random.h"

using namespace metal;

kernel void pinhole_camera_generate_rays(
    device RayData *ray_buffer [[buffer(0)]],
    constant CameraData &camera_data [[buffer(1)]],
    constant FrameData &frame_data [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
    
    auto w = frame_data.size.x;
    auto h = frame_data.size.y;
    
    if (tid.x < w && tid.y < h) {
        
        auto seed = (tea<4>(tid.x, tid.y) + frame_data.index) << 8u;
        
        auto z = camera_data.near_plane;
        auto half_sensor_width = tan(0.5f * camera_data.fov) * z;
        auto half_sensor_height = half_sensor_width * h / w;
        
        auto px = static_cast<float>(tid.x) + halton(seed);
        auto py = static_cast<float>(tid.y) + halton(seed);
    
        auto x = (1.0f - px / w * 2.0f) * half_sensor_width;
        auto y = (1.0f - py / h * 2.0f) * half_sensor_height;
        
        RayData ray{};
        ray.origin = camera_data.position;
        ray.direction = normalize(x * camera_data.left + y * camera_data.up + z * camera_data.front);
        ray.min_distance = 0.0f;
        ray.max_distance = INFINITY;
        ray.throughput = PackedVec3f{1.0f, 1.0f, 1.0f};
        ray.seed = seed;
        ray.radiance = PackedVec3f{0.0f, 0.0f, 0.0f};
        ray.depth = 0;
        ray.pixel = Vec2f{px, py};
        
        ray_buffer[tid.y * w + tid.x] = ray;
    }
}
