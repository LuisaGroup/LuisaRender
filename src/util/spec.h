//
// Created by Mike Smith on 2022/1/19.
//

#pragma once

#include <core/basic_types.h>
#include <core/stl.h>
#include <dsl/syntax.h>
#include <runtime/command_buffer.h>
#include <util/colorspace.h>

namespace luisa::render {

constexpr auto visible_wavelength_min = 360.0f;
constexpr auto visible_wavelength_max = 830.0f;

constexpr auto rgb_spectrum_peak_wavelengths =
    make_float3(602.785f, 539.285f, 445.772f);

using compute::ArrayVar;
using compute::BindlessArray;
using compute::Bool;
using compute::CommandBuffer;
using compute::Constant;
using compute::Expr;
using compute::Float;
using compute::Float3;
using compute::Float4;
using compute::Local;
using compute::VolumeView;

class DenselySampledSpectrum {

public:
    static constexpr auto sample_count =
        static_cast<uint>(visible_wavelength_max - visible_wavelength_min) + 1u;

private:
    Constant<float> _values;

private:
    explicit DenselySampledSpectrum(luisa::span<const float, sample_count> values) noexcept
        : _values{values} {}

public:
    [[nodiscard]] static const DenselySampledSpectrum &cie_x() noexcept;
    [[nodiscard]] static const DenselySampledSpectrum &cie_y() noexcept;
    [[nodiscard]] static const DenselySampledSpectrum &cie_z() noexcept;
    [[nodiscard]] static const DenselySampledSpectrum &cie_illum_d65() noexcept;
    [[nodiscard]] Float sample(Expr<float> lambda) const noexcept;
    [[nodiscard]] static float cie_y_integral() noexcept;
};

class SampledSpectrum {

private:
    Local<float> _samples;

public:
    explicit SampledSpectrum(size_t n, Expr<float> value = 0.f) noexcept : _samples{n} {
        for (auto i = 0u; i < n; i++) { _samples[i] = value; }
    }
    auto &operator=(Expr<float> value) noexcept {
        for (auto i = 0u; i < dimension(); i++) { _samples[i] = value; }
        return *this;
    }
    [[nodiscard]] uint dimension() const noexcept {
        return static_cast<uint>(_samples.size());
    }
    [[nodiscard]] Local<float> &values() noexcept { return _samples; }
    [[nodiscard]] const Local<float> &values() const noexcept { return _samples; }
    [[nodiscard]] Float &operator[](Expr<uint> i) noexcept { return _samples[i]; }
    [[nodiscard]] Float operator[](Expr<uint> i) const noexcept { return _samples[i]; }
    template<typename F>
    [[nodiscard]] auto map(F &&f) const noexcept {
        SampledSpectrum s{_samples.size()};
        for (auto i = 0u; i < dimension(); i++) { s[i] = f(i, Expr{(*this)[i]}); }
        return s;
    }
    template<typename T, typename F>
    [[nodiscard]] auto reduce(T &&initial, F &&f) const noexcept {
        using compute::def;
        auto r = def(std::forward<T>(initial));
        for (auto i = 0u; i < dimension(); i++) { r = f(Expr{r}, i, Expr{(*this)[i]}); }
        return r;
    }
    [[nodiscard]] auto sum() const noexcept {
        return reduce(0.f, [](auto r, auto, auto x) noexcept { return r + x; });
    }
    [[nodiscard]] auto average() const noexcept {
        return sum() * static_cast<float>(1.0 / dimension());
    }
    template<typename F>
    [[nodiscard]] auto any(F &&f) const noexcept {
        return reduce(false, [&f](auto ans, auto, auto value) noexcept { return ans | f(value); });
    }
    template<typename F>
    [[nodiscard]] auto all(F &&f) const noexcept {
        return reduce(true, [&f](auto ans, auto, auto value) noexcept { return ans & f(value); });
    }
    template<typename F>
    [[nodiscard]] auto none(F &&f) const noexcept { return !any(std::forward<F>(f)); }

    [[nodiscard]] auto operator+() const noexcept {
        return map([](auto, auto s) noexcept { return s; });
    }
    [[nodiscard]] auto operator-() const noexcept {
        return map([](auto, auto s) noexcept { return -s; });
    }
    [[nodiscard]] auto isnan() const noexcept {
        return map([](auto, auto s) noexcept { return compute::isnan(s); });
    }
    [[nodiscard]] auto abs() const noexcept {
        return map([](auto, auto s) noexcept { return compute::abs(s); });
    }

#define LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(op)                                    \
    [[nodiscard]] auto operator op(Expr<float> rhs) const noexcept {                        \
        return map([rhs](auto, const auto &lvalue) { return lvalue op rhs; });              \
    }                                                                                       \
    [[nodiscard]] auto operator op(const SampledSpectrum &rhs) const noexcept {             \
        return map([&rhs](auto i, const auto &lvalue) { return lvalue op rhs[i]; });        \
    }                                                                                       \
    friend auto operator op(Expr<float> lhs, const SampledSpectrum &rhs) noexcept {         \
        return rhs.map([lhs](auto, const auto &rvalue) noexcept { return lhs op rvalue; }); \
    }                                                                                       \
    auto &operator op##=(Expr<float> rhs) noexcept {                                        \
        for (auto i = 0u; i < dimension(); i++) { (*this)[i] op## = rhs; }                  \
        return *this;                                                                       \
    }                                                                                       \
    auto &operator op##=(const SampledSpectrum &rhs) noexcept {                             \
        for (auto i = 0u; i < dimension(); i++) { (*this)[i] op## = rhs[i]; }               \
        return *this;                                                                       \
    }
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(+)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(-)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(*)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(/)
#undef LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP

public:
    static SampledSpectrum ite(const SampledSpectrum &p, const SampledSpectrum &t, const SampledSpectrum &f) noexcept;
    static SampledSpectrum ite(Expr<bool> p, const SampledSpectrum &t, const SampledSpectrum &f) noexcept;
};

SampledSpectrum any_nan2zero(const SampledSpectrum &t) noexcept;

}// namespace luisa::render
