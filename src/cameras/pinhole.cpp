//
// Created by Mike Smith on 2020/9/14.
//

#include <compute/dsl.h>
#include <render/camera.h>

namespace luisa::render::camera {

using namespace compute;
using namespace compute::dsl;

class PinholeCamera : public Camera {

private:
    float3 _position;
    float3 _front{};
    float3 _up{};
    float3 _left{};
    float4x4 _camera_to_world{};
    float2 _sensor_size{};
    float _near_plane;

private:
    void _generate_rays(Pipeline &pipeline,
                        float4x4 camera_to_world,
                        Sampler &sampler [[maybe_unused]],
                        const BufferView<float2> &pixel_positions,
                        BufferView<Ray> &rays,
                        BufferView<float3> &throughputs) override {
        
        auto resolution = film()->resolution();
        auto pixel_count = resolution.x * resolution.y;
        constexpr auto threadgroup_size = 256u;
        
        pipeline << [this, camera_to_world] { _camera_to_world = camera_to_world; };
        
        auto kernel = device()->compile_kernel("pinhole_camara_generate_rays", [&] {
            auto tid = thread_id();
            If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                Var pixel = pixel_positions[tid];
                Var p_film = (dsl::make_float2(0.5f) - pixel / dsl::make_float2(resolution)) * dsl::make_float2(_sensor_size) * 0.5f;
                Var o_world = make_float3(uniform(&_camera_to_world) * dsl::make_float4(_position, 1.0f));
                Var p_film_world = make_float3(uniform(&_camera_to_world) *
                                               make_float4(p_film.x() * _left + p_film.y() * _up + _near_plane * _front + _position, 1.0f));
                Var d = normalize(p_film_world - o_world);
                
                Var<Ray> ray;
                ray.origin_x() = o_world.x();
                ray.origin_y() = o_world.y();
                ray.origin_z() = o_world.z();
                ray.min_distance() = 1e-3f;
                ray.direction_x() = d.x();
                ray.direction_y() = d.y();
                ray.direction_z() = d.z();
                ray.max_distance() = 1e3f;
                
                rays[tid] = ray;
                throughputs[tid] = dsl::make_float3(1.0f);
            };
        });
        
        pipeline << kernel.parallelize(pixel_count, threadgroup_size);
    }

public:
    PinholeCamera(Device *device, const ParameterSet &params)
        : Camera{device, params},
          _position{params["position"].parse_float3()},
          _near_plane{params["near_plane"].parse_float_or_default(0.1f)} {
        
        auto fov = math::radians(params["fov"].parse_float_or_default(35.0f));
        auto res = make_float2(film()->resolution());
        auto sensor_height = 2.0f * _near_plane * tan(0.5f * fov);
        auto sensor_width = sensor_height * res.x / res.y;
        _sensor_size = make_float2(sensor_width, sensor_height);
        
        auto up = params["up"].parse_float3_or_default(make_float3(0.0f, 1.0f, 0.0f));
        _front = normalize(params["target"].parse_float3() - _position);
        _left = normalize(cross(up, _front));
        _up = normalize(cross(_front, _left));
    }
    
};

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::render::camera::PinholeCamera)
