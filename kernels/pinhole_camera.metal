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
        
        auto sensor_x = 1.0f - (static_cast<float>(tid.x) + halton(seed)) / w * 2.0f;
        auto sensor_y = 1.0f - (static_cast<float>(tid.y) + halton(seed)) / h * 2.0f;
        
        auto x = sensor_x * half_sensor_width;
        auto y = sensor_y * half_sensor_height;
        
        auto w_world = normalize(x * camera_data.left + y * camera_data.up + z * camera_data.front);
        auto o_world = camera_data.position;
        
        RayData ray{};
        ray.origin = o_world;
        ray.direction = w_world;
        ray.min_distance = 1e-3f;
        ray.max_distance = INFINITY;
        ray.throughput = PackedVec3f{1.0f, 1.0f, 1.0f};
        ray.seed = seed;
        ray.radiance = PackedVec3f{0.0f, 0.0f, 0.0f};
        ray.depth = 0;
        
        ray_buffer[tid.y * w + tid.x] = ray;
    }
}
