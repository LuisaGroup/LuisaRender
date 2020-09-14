//
// Created by Mike Smith on 2020/9/14.
//

#include <compute/dsl.h>
#include "filter.h"

namespace luisa::render {

using namespace luisa::compute;
using namespace luisa::compute::dsl;

void SeparableFilter::_importance_sample_pixel_positions(
    Pipeline &pipeline,
    Sampler &sampler,
    BufferView<float2> &position_buffer,
    BufferView<float> &weight_buffer) {
    
    // compute lut
    constexpr auto inv_table_size = 1.0f / lookup_table_size;
    
    std::array<float, lookup_table_size> weight_table{};
    std::array<float, lookup_table_size> cdf_table{};
    
    auto abs_sum = 0.0f;
    for (auto i = 0u; i < lookup_table_size; i++) {
        auto offset = (static_cast<float>(i) * inv_table_size * 2.0f - 1.0f) * radius();
        auto w = _weight_1d(offset);
        weight_table[i] = w;
        cdf_table[i] = (abs_sum += std::abs(w));
    }
    auto inv_sum = 1.0f / abs_sum;
    for (float &cdf : cdf_table) { cdf *= inv_sum; }
    
    auto absolute_volume = 0.0f;
    auto signed_volume = 0.0f;
    for (float u : weight_table) {
        for (float v : weight_table) {
            signed_volume += u * v;
            absolute_volume += std::abs(u * v);
        }
    }
    auto scale = absolute_volume / signed_volume;
    
    pipeline << sampler.generate_samples(2u);
    
    auto pixel_count = sampler.sample_texture().width() * sampler.sample_texture().height();
    constexpr auto threadgroup_size = 256u;
    
    // compile kernel
    auto kernel = device()->compile_kernel("filter_importance_sampling", [&] {
        
        Var weight = weight_table;
        Var cdf = cdf_table;
        
        auto sample_1d = [&](Expr<float> u) {
            
            Var p = 0u;
            Var count = static_cast<int>(lookup_table_size);
            While (count > 0) {
                Var step = count / 2;
                Var mid = p + step;
                If (weight[mid] < u) {
                    p = mid + 1;
                    count -= step + 1;
                } Else {
                    count = step;
                };
            };
            
            Var lb = dsl::clamp(p, 0u, lookup_table_size - 1u);
            Var cdf_lower = cdf[lb];
            Var cdf_upper = select(lb == lookup_table_size - 1u, 1.0f, cdf[lb + 1u]);
            Var offset = dsl::clamp((cast<float>(lb) + (u - cdf_lower) / (cdf_upper - cdf_lower)) * inv_table_size, 0.0f, 1.0f);
    
            constexpr auto weight_table_size_float = static_cast<float>(lookup_table_size);
            Var index_w = offset * weight_table_size_float;
            Var index_w_lower = floor(index_w);
            Var index_w_upper = ceil(index_w);
            Var w = lerp(
                cdf[cast<uint>(index_w_lower)],
                select(index_w_upper >= weight_table_size_float, 0.0f, weight[cast<uint>(index_w_upper)]),
                index_w - index_w_lower);
            
            return std::make_pair(offset * 2.0f - 1.0f, select(w >= 0.0f, 1.0f, 0.0f));
        };
        
        auto tid = thread_id();
        If (pixel_count % threadgroup_size == 0u || tid < pixel_count) {
            Var px = tid % sampler.sample_texture().width();
            Var py = tid / sampler.sample_texture().width();
            Var u = make_float2(sampler.sample_texture().read(make_uint2(px, py)));
            auto[dx, wx] = sample_1d(u.x());
            auto[dy, wy] = sample_1d(u.y());
            position_buffer[tid] = make_float2(px, py) + 0.5f + make_float2(dx, dy) * radius();
            weight_buffer[tid] = wx * wy * scale;
        };
    });
    
    pipeline << kernel.parallelize(pixel_count, threadgroup_size);
}

}
