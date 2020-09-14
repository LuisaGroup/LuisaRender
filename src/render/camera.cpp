//
// Created by Mike Smith on 2020/9/14.
//

#include <compute/dsl.h>
#include "camera.h"

namespace luisa::render {

using namespace luisa::compute;
using namespace luisa::compute::dsl;

void Camera::_generate_pixel_positions_without_filter(
    Pipeline &pipeline,
    Sampler &sampler,
    BufferView<float2> &pixel_position_buffer,
    BufferView<float> &filter_weight_buffer) {
    
    pipeline << sampler.generate_samples(2u);
    constexpr auto threadgroup_size = 256u;
    auto pixel_count = sampler.sample_texture().width() * sampler.sample_texture().height();
    auto kernel = device()->compile_kernel("generate_pixel_positions_without_filtering", [&] {
        auto tid = thread_id();
        If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
            Var px = tid % sampler.sample_texture().width();
            Var py = tid / sampler.sample_texture().width();
            Var u = make_float2(sampler.sample_texture().read(make_uint2(px, py)));
            pixel_position_buffer[tid] = make_float2(px, py) + u;
            filter_weight_buffer[tid] = 1.0f;
        };
    });
    pipeline << kernel.parallelize(pixel_count, threadgroup_size);
}

}
