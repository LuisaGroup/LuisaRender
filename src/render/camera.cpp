//
// Created by Mike Smith on 2020/9/14.
//

#include <compute/dsl.h>
#include "camera.h"

namespace luisa::render {

using namespace luisa::compute;
using namespace luisa::compute::dsl;

std::function<void(Pipeline &pipeline)> Camera::generate_rays(float time, Sampler &sampler) {
    
    static constexpr auto threadgroup_size = 256u;
    auto pixel_count = _film->resolution().x * _film->resolution().y;
    
    if (_generate_rays_kernel.empty()) {
        _generate_rays_kernel = device()->compile_kernel("camera_generate_rays", [&] {
            auto tid = thread_id();
            If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
                Var u = sampler.generate_4d_sample(tid);
                Var p = make_uint2(tid % _film->resolution().x, tid / _film->resolution().x);
                auto[px, px_w] = _filter == nullptr ?
                                 std::make_pair(make_float2(p) + make_float2(u), Expr{1.0f}) :
                                 _filter->importance_sample_pixel_position(p, make_float2(u));
                Var pixel_position = px;
                _pixel_position_buffer[tid] = pixel_position;
                _filter_weight_buffer[tid] = px_w;
                
                auto [ray, throughput] = _generate_rays(uniform(&_camera_to_world), make_float2(u.z(), u.w()), pixel_position);
                _camera_ray_buffer[tid] = ray;
                _throughput_buffer[tid] = throughput;
            };
        });
    }
    
    return [this, time, pixel_count](Pipeline &pipeline) {
        pipeline << [this, time] { _camera_to_world = _transform == nullptr ? make_float4x4(1.0f) : _transform->matrix(time); }
                 << _generate_rays_kernel.parallelize(pixel_count, threadgroup_size);
    };
}

}
