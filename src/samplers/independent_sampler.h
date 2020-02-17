//
// Created by Mike Smith on 2020/2/14.
//

#pragma once

#include <core/data_types.h>
#include <core/mathematics.h>

namespace luisa::independent_sampler {

LUISA_CONSTANT_SPACE constexpr auto PCG32_DEFAULT_STATE = 0x853c49e6748fea9bull;
LUISA_CONSTANT_SPACE constexpr auto PCG32_MULT = 0x5851f42d4c957f2dull;
LUISA_CONSTANT_SPACE constexpr auto ONE_MINUS_EPSILON = 0x1.fffffep-1f;

template<uint N>
LUISA_DEVICE_CALLABLE inline uint tea(uint v0, uint v1) {
    auto s0 = 0u;
    for (auto n = 0u; n < N; n++) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
        v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
    }
    return v0;
}

struct SamplerState {
    uint64_t seed;
    uint64_t inc;
};

LUISA_DEVICE_CALLABLE inline uint generate_uniform_uint(LUISA_THREAD_SPACE SamplerState &state) noexcept {
    auto old = state.seed;
    state.seed = old * PCG32_MULT + state.inc;
    auto xor_shifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
    auto rot = static_cast<uint32_t>(old >> 59u);
    return (xor_shifted >> rot) | (xor_shifted << ((~rot + 1u) & 31u));
}

LUISA_DEVICE_CALLABLE inline SamplerState make_sampler_state(uint64_t init_seq) noexcept {
    auto seed = 0ull;
    auto inc = (init_seq << 1u) | 1ull;
    SamplerState state{seed, inc};
    generate_uniform_uint(state);
    state.seed += PCG32_DEFAULT_STATE;
    generate_uniform_uint(state);
    return state;
}

LUISA_DEVICE_CALLABLE inline void reset_states(
    uint2 film_resolution,
    LUISA_DEVICE_SPACE SamplerState *sampler_state_buffer,
    uint tid) {
    
    if (tid < film_resolution.x * film_resolution.y) {
        sampler_state_buffer[tid] = make_sampler_state(tea<5>(tid % film_resolution.x, tid / film_resolution.x));
    }
    
}

template<uint dimension>
LUISA_DEVICE_CALLABLE inline void generate_samples(
    LUISA_DEVICE_SPACE SamplerState *sampler_state_buffer,
    LUISA_DEVICE_SPACE const uint *ray_queue,
    uint ray_count,
    LUISA_DEVICE_SPACE float *sample_buffer,
    uint tid) {
    
    if (tid < ray_count) {
        auto ray_index = ray_queue[tid];
        auto state = sampler_state_buffer[ray_index];
        for (auto i = 0u; i < dimension; i++) {
            sample_buffer[tid * dimension + i] = min(ONE_MINUS_EPSILON, generate_uniform_uint(state) * 0x1p-32f);
        }
        sampler_state_buffer[ray_index] = state;
    }
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <core/sampler.h>

namespace luisa {

class IndependentSampler : public Sampler {

protected:
    std::unique_ptr<Kernel> _reset_states_kernel;
    std::unique_ptr<Kernel> _generate_1d_samples_kernel;
    std::unique_ptr<Kernel> _generate_2d_samples_kernel;
    std::unique_ptr<Kernel> _generate_3d_samples_kernel;
    std::unique_ptr<Kernel> _generate_4d_samples_kernel;
    std::unique_ptr<Buffer<independent_sampler::SamplerState>> _state_buffer;
    
    void _generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float> sample_buffer) override;
    void _generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float2> sample_buffer) override;
    void _generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float3> sample_buffer) override;
    void _generate_samples(KernelDispatcher &dispatch, BufferView<uint> ray_queue_buffer, BufferView<uint> ray_count_buffer, BufferView<float4> sample_buffer) override;

public:
    IndependentSampler(Device *device, const ParameterSet &parameter_set);
    void reset_states(KernelDispatcher &dispatch, uint2 film_resolution) override;
};

LUISA_REGISTER_NODE_CREATOR("Independent", IndependentSampler);

}

#endif
