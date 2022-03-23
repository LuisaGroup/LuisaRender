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
    explicit SampledSpectrum(size_t n, Expr<float> s = 0.f) noexcept : _samples{n} {
        for (auto i = 0u; i < n; i++) { _samples[i] = s; }
    }
    [[nodiscard]] auto dimension() const noexcept { return _samples.size(); }
    [[nodiscard]] auto &values() noexcept { return _samples; }
    [[nodiscard]] auto &values() const noexcept { return _samples; }
    [[nodiscard]] Float &at(Expr<uint> i) noexcept { return _samples[i]; }
    [[nodiscard]] Float at(Expr<uint> i) const noexcept { return _samples[i]; }
    [[nodiscard]] Float &operator[](Expr<uint> i) noexcept { return at(i); }
    [[nodiscard]] Float operator[](Expr<uint> i) const noexcept { return at(i); }
    template<typename F>
    void for_each(F &&f) noexcept {
        for (auto i = 0u; i < dimension(); i++) { f(i, (*this)[i]); }
    }
    template<typename F>
    void for_each(F &&f) const noexcept {
        for (auto i = 0u; i < dimension(); i++) { f(i, Expr{(*this)[i]}); }
    }
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
        return reduce(false, [&f](auto r, auto, auto s) noexcept { return r | f(s); });
    }
    template<typename F>
    [[nodiscard]] auto all(F &&f) const noexcept {
        return reduce(true, [&f](auto r, auto, auto s) noexcept { return r & f(s); });
    }
    template<typename F>
    [[nodiscard]] auto none(F &&f) const noexcept { return !any(std::forward<F>(f)); }

    [[nodiscard]] auto operator+() const noexcept {
        return map([](auto, auto s) noexcept { return s; });
    }
    [[nodiscard]] auto operator-() const noexcept {
        return map([](auto, auto s) noexcept { return -s; });
    }
#define LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(op)                            \
    [[nodiscard]] auto operator op(Expr<float> rhs) const noexcept {                \
        return map([rhs](auto, auto lhs) { return lhs op rhs; });                   \
    }                                                                               \
    [[nodiscard]] auto operator op(const SampledSpectrum &rhs) const noexcept {     \
        return map([&rhs](auto i, auto lhs) { return lhs op rhs[i]; });             \
    }                                                                               \
    friend auto operator op(Expr<float> lhs, const SampledSpectrum &rhs) noexcept { \
        return rhs.map([lhs](auto, auto r) noexcept { return lhs op r; });          \
    }                                                                               \
    SampledSpectrum &operator op##=(Expr<float> rhs) noexcept {                     \
        for (auto i = 0u; i < dimension(); i++) { (*this)[i] op## = rhs; }          \
        return *this;                                                               \
    }                                                                               \
    SampledSpectrum &operator op##=(const SampledSpectrum &rhs) noexcept {          \
        for (auto i = 0u; i < dimension(); i++) { (*this)[i] op## = rhs[i]; }       \
        return *this;                                                               \
    }
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(+)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(-)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(*)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(/)
#undef LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP
};

}// namespace luisa::render
