//
// Created by Mike Smith on 2022/9/14.
//

#include <util/spec.h>
#include <base/spd.h>
#include <base/pipeline.h>

namespace luisa::render {

SPD::SPD(Pipeline &pipeline, uint buffer_id, float sample_interval) noexcept
    : _pipeline{pipeline}, _buffer_id{buffer_id}, _sample_interval{sample_interval} {}

static inline auto densely_sampled_spectrum_integral(uint t, const float *spec) noexcept {
    auto sum = 0.0;
    auto tt = static_cast<float>(t);
    auto n = (visible_wavelength_max - visible_wavelength_min) / tt;
    LUISA_ASSERT(n == std::floor(n), "Invalid SPD sample interval.");
    auto nn = static_cast<uint>(n) + 1u;
    for (auto i = 0u; i < nn - 1u; i++) {
        sum += 0.5f * (cie_y_samples[i * t] + cie_y_samples[(i + 1u) * t]);
    }
    return static_cast<float>(sum * tt);
}

static inline auto downsample_densely_sampled_spectrum(uint t, const float *spec) noexcept {
    auto n = (visible_wavelength_max - visible_wavelength_min) / static_cast<float>(t);
    LUISA_ASSERT(n == std::floor(n), "Invalid SPD sample interval.");
    auto nn = static_cast<uint>(n) + 1u;
    luisa::vector<float> samples(nn);
    for (auto x = 0u; x < nn; x++) { samples[x] = spec[x * t]; }
    return samples;
}

static constexpr auto spd_lut_interval = 5u;

SPD SPD::create_cie_x(Pipeline &pipeline, CommandBuffer &cb) noexcept {
    auto buffer = pipeline.register_named_id(
        "__internal_spd_cie_x", [&pipeline, &cb] {
            auto s = downsample_densely_sampled_spectrum(
                spd_lut_interval, cie_x_samples.data());
            auto [view, index] = pipeline.bindless_arena_buffer<float>(s.size());
            cb << view.copy_from(s.data()) << compute::commit();
            return index;
        });
    return {pipeline, buffer, spd_lut_interval};
}

SPD SPD::create_cie_y(Pipeline &pipeline, CommandBuffer &cb) noexcept {
    auto buffer = pipeline.register_named_id(
        "__internal_spd_cie_y", [&pipeline, &cb] {
            auto s = downsample_densely_sampled_spectrum(
                spd_lut_interval, cie_y_samples.data());
            auto [view, index] = pipeline.bindless_arena_buffer<float>(s.size());
            cb << view.copy_from(s.data()) << compute::commit();
            return index;
        });
    return {pipeline, buffer, spd_lut_interval};
}

SPD SPD::create_cie_z(Pipeline &pipeline, CommandBuffer &cb) noexcept {
    auto buffer = pipeline.register_named_id(
        "__internal_spd_cie_z", [&pipeline, &cb] {
            auto s = downsample_densely_sampled_spectrum(
                spd_lut_interval, cie_z_samples.data());
            auto [view, index] = pipeline.bindless_arena_buffer<float>(s.size());
            cb << view.copy_from(s.data()) << compute::commit();
            return index;
        });
    return {pipeline, buffer, spd_lut_interval};
}

SPD SPD::create_cie_d65(Pipeline &pipeline, CommandBuffer &cb) noexcept {
    auto buffer = pipeline.register_named_id(
        "__internal_spd_cie_d65", [&pipeline, &cb] {
            auto s = downsample_densely_sampled_spectrum(
                spd_lut_interval, cie_d65_samples.data());
            auto [view, index] = pipeline.bindless_arena_buffer<float>(s.size());
            cb << view.copy_from(s.data()) << compute::commit();
            return index;
        });
    return {pipeline, buffer, spd_lut_interval};
}

float SPD::cie_y_integral() noexcept {
    static auto integral = densely_sampled_spectrum_integral(
        spd_lut_interval, cie_y_samples.data());
    return integral;
}

Float SPD::sample(Expr<float> lambda) const noexcept {
    using namespace luisa::compute;
    auto t = (clamp(lambda, visible_wavelength_min, visible_wavelength_max) - visible_wavelength_min) / _sample_interval;
    auto sample_count = static_cast<uint>((visible_wavelength_max - visible_wavelength_min) / _sample_interval) + 1u;
    auto i = cast<uint>(min(t, static_cast<float>(sample_count - 2u)));
    auto s0 = _pipeline.buffer<float>(_buffer_id).read(i);
    auto s1 = _pipeline.buffer<float>(_buffer_id).read(i + 1u);
    return lerp(s0, s1, fract(t));
}

}// namespace luisa::render
