//
// Created by Mike Smith on 2022/1/19.
//

#pragma once

#include <core/basic_types.h>
#include <core/stl.h>
#include <core/logging.h>
#include <dsl/syntax.h>
#include <util/command_buffer.h>
#include <util/colorspace.h>

namespace luisa::render {

constexpr auto visible_wavelength_min = 360.0f;
constexpr auto visible_wavelength_max = 830.0f;

constexpr auto fraunhofer_wavelengths =
    make_float3(656.27f, 587.56f, 486.13f);

constexpr auto rgb_spectrum_peak_wavelengths =
    make_float3(602.785f, 539.285f, 445.772f);

constexpr auto cie_sample_count = static_cast<uint>(
    visible_wavelength_max - visible_wavelength_min + 1.0f);
static_assert(cie_sample_count == 471u);

LUISA_IMPORT_API const std::array<float, cie_sample_count> cie_x_samples;
LUISA_IMPORT_API const std::array<float, cie_sample_count> cie_y_samples;
LUISA_IMPORT_API const std::array<float, cie_sample_count> cie_z_samples;
LUISA_IMPORT_API const std::array<float, cie_sample_count> cie_d65_samples;

using compute::ArrayVar;
using compute::BindlessArray;
using compute::Bool;
using compute::Constant;
using compute::Expr;
using compute::Float;
using compute::Float3;
using compute::Float4;
using compute::Local;
using compute::VolumeView;

template<typename X, typename C0>
[[nodiscard]] inline auto &polynomial(const X &x, const C0 &c0) noexcept { return c0; }

template<typename X, typename C0, typename... C>
[[nodiscard]] inline auto polynomial(const X &x, const C0 &c0, const C &...c) noexcept {
    return x * polynomial(x, c...) + c0;
}

class SampledSpectrum {

private:
    Local<float> _samples;

public:
    SampledSpectrum(uint n, Expr<float> value) noexcept : _samples{n} {
        compute::outline([&] {
            for (auto i = 0u; i < n; i++) { _samples[i] = value; }
        });
    }
    explicit SampledSpectrum(uint n) noexcept : SampledSpectrum{n, 0.f} {}
    explicit SampledSpectrum(Expr<float> value) noexcept : SampledSpectrum{1u, value} {}
    explicit SampledSpectrum(float value) noexcept : SampledSpectrum{1u, value} {}
    auto &operator=(Expr<float> value) noexcept {
        compute::outline([&] {
            for (auto i = 0u; i < dimension(); i++) { _samples[i] = value; }
        });
        return *this;
    }
    [[nodiscard]] uint dimension() const noexcept {
        return static_cast<uint>(_samples.size());
    }
    auto &operator=(const SampledSpectrum &rhs) noexcept {
        LUISA_ASSERT(rhs.dimension() == 1u || dimension() == rhs.dimension(),
                     "Invalid spectrum dimensions for operator=: {} vs {}.",
                     dimension(), rhs.dimension());
        compute::outline([&] {
            for (auto i = 0u; i < dimension(); i++) { _samples[i] = rhs[i]; }
        });
        return *this;
    }
    [[nodiscard]] Local<float> &values() noexcept { return _samples; }
    [[nodiscard]] const Local<float> &values() const noexcept { return _samples; }
    [[nodiscard]] Float &operator[](Expr<uint> i) noexcept {
        return dimension() == 1u ? _samples[0u] : _samples[i];
    }
    [[nodiscard]] Float operator[](Expr<uint> i) const noexcept {
        return dimension() == 1u ? _samples[0u] : _samples[i];
    }
    template<typename F>
    [[nodiscard]] auto map(F &&f) const noexcept {
        SampledSpectrum s{dimension()};
        compute::outline([&] {
            for (auto i = 0u; i < dimension(); i++) {
                if constexpr (std::invocable<F, Expr<float>>) {
                    s[i] = f(Expr{(*this)[i]});
                } else {
                    s[i] = f(i, Expr{(*this)[i]});
                }
            }
        });
        return s;
    }
    template<typename T, typename F>
    [[nodiscard]] auto reduce(T &&initial, F &&f) const noexcept {
        using compute::def;
        auto r = def(std::forward<T>(initial));
        compute::outline([&] {
            for (auto i = 0u; i < dimension(); i++) {
                if constexpr (std::invocable<F, Expr<compute::expr_value_t<decltype(r)>>, Expr<float>>) {
                    r = f(r, Expr{(*this)[i]});
                } else {
                    r = f(Expr{r}, i, Expr{(*this)[i]});
                }
            }
        });
        return r;
    }
    [[nodiscard]] auto sum() const noexcept {
        return reduce(0.f, [](auto r, auto x) noexcept { return r + x; });
    }
    [[nodiscard]] auto max() const noexcept {
        return reduce(0.f, [](auto r, auto x) noexcept {
            return luisa::compute::max(r, x);
        });
    }
    [[nodiscard]] auto min() const noexcept {
        return reduce(std::numeric_limits<float>::max(), [](auto r, auto x) noexcept {
            return luisa::compute::min(r, x);
        });
    }
    [[nodiscard]] auto average() const noexcept {
        return sum() * static_cast<float>(1.0 / dimension());
    }
    template<typename F>
    [[nodiscard]] auto any(F &&f) const noexcept {
        return reduce(false, [&f](auto ans, auto value) noexcept { return ans | f(value); });
    }
    template<typename F>
    [[nodiscard]] auto all(F &&f) const noexcept {
        return reduce(true, [&f](auto ans, auto value) noexcept { return ans & f(value); });
    }
    [[nodiscard]] auto is_zero() const noexcept {
        return all([](auto x) noexcept { return x == 0.f; });
    }
    template<typename F>
    [[nodiscard]] auto none(F &&f) const noexcept { return !any(std::forward<F>(f)); }

    [[nodiscard]] auto operator+() const noexcept {
        return map([](auto s) noexcept { return s; });
    }
    [[nodiscard]] auto operator-() const noexcept {
        return map([](auto s) noexcept { return -s; });
    }

#define LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(op)                                           \
    [[nodiscard]] auto operator op(Expr<float> rhs) const noexcept {                               \
        return map([rhs](const auto &lvalue) { return lvalue op rhs; });                           \
    }                                                                                              \
    [[nodiscard]] auto operator op(const SampledSpectrum &rhs) const noexcept {                    \
        LUISA_ASSERT(dimension() == 1u || rhs.dimension() == 1u || dimension() == rhs.dimension(), \
                     "Invalid sampled spectrum dimension for operator" #op ": {} vs {}.",          \
                     dimension(), rhs.dimension());                                                \
        SampledSpectrum s{std::max(dimension(), rhs.dimension())};                                 \
        compute::outline([&] {                                                                     \
            for (auto i = 0u; i < s.dimension(); i++) { s[i] = (*this)[i] op rhs[i]; }             \
        });                                                                                        \
        return s;                                                                                  \
    }                                                                                              \
    [[nodiscard]] friend auto operator op(Expr<float> lhs, const SampledSpectrum &rhs) noexcept {  \
        return rhs.map([lhs](const auto &rvalue) noexcept { return lhs op rvalue; });              \
    }                                                                                              \
    auto &operator op##=(Expr<float> rhs) noexcept {                                               \
        compute::outline([&] {                                                                     \
            for (auto i = 0u; i < dimension(); i++) { (*this)[i] op## = rhs; }                     \
        });                                                                                        \
        return *this;                                                                              \
    }                                                                                              \
    auto &operator op##=(const SampledSpectrum &rhs) noexcept {                                    \
        LUISA_ASSERT(rhs.dimension() == 1u || dimension() == rhs.dimension(),                      \
                     "Invalid sampled spectrum dimension for operator" #op "=: {} vs {}.",         \
                     dimension(), rhs.dimension());                                                \
        if (rhs.dimension() == 1u) { return *this op## = rhs[0u]; }                                \
        compute::outline([&] {                                                                     \
            for (auto i = 0u; i < dimension(); i++) { (*this)[i] op## = rhs[i]; }                  \
        });                                                                                        \
        return *this;                                                                              \
    }
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(+)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(-)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(*)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(/)
#undef LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP

#define LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_COMPARISON_OP(op)                                      \
    [[nodiscard]] auto operator op(Expr<float> rhs) const noexcept {                              \
        return map([rhs](const auto &lvalue) { return lvalue op rhs; });                          \
    }                                                                                             \
    [[nodiscard]] auto operator op(const SampledSpectrum &rhs) const noexcept {                   \
        return map([&rhs](auto i, const auto &lvalue) { return lvalue op rhs[i]; });              \
    }                                                                                             \
    [[nodiscard]] friend auto operator op(Expr<float> lhs, const SampledSpectrum &rhs) noexcept { \
        return rhs.map([lhs](const auto &rvalue) noexcept { return lhs op rvalue; });             \
    }
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_COMPARISON_OP(>)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_COMPARISON_OP(>=)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_COMPARISON_OP(<)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_COMPARISON_OP(<=)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_COMPARISON_OP(==)
#undef LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_COMPARISON_OP
};

// functions
using luisa::clamp;
using luisa::max;
using luisa::min;
using luisa::compute::clamp;
using luisa::compute::ite;
using luisa::compute::max;
using luisa::compute::min;

[[nodiscard]] SampledSpectrum ite(const SampledSpectrum &p, const SampledSpectrum &t, const SampledSpectrum &f) noexcept;
[[nodiscard]] SampledSpectrum ite(const SampledSpectrum &p, Expr<float> t, const SampledSpectrum &f) noexcept;
[[nodiscard]] SampledSpectrum ite(const SampledSpectrum &p, const SampledSpectrum &t, Expr<float> f) noexcept;
[[nodiscard]] SampledSpectrum ite(const SampledSpectrum &p, Expr<float> t, Expr<float> f) noexcept;
[[nodiscard]] SampledSpectrum ite(Expr<bool> p, const SampledSpectrum &t, const SampledSpectrum &f) noexcept;
[[nodiscard]] SampledSpectrum ite(Expr<bool> p, Expr<float> t, const SampledSpectrum &f) noexcept;
[[nodiscard]] SampledSpectrum ite(Expr<bool> p, const SampledSpectrum &t, Expr<float> f) noexcept;
[[nodiscard]] SampledSpectrum max(const SampledSpectrum &a, Expr<float> b) noexcept;
[[nodiscard]] SampledSpectrum max(Expr<float> a, const SampledSpectrum &b) noexcept;
[[nodiscard]] SampledSpectrum max(const SampledSpectrum &a, const SampledSpectrum &b) noexcept;
[[nodiscard]] SampledSpectrum min(const SampledSpectrum &a, Expr<float> b) noexcept;
[[nodiscard]] SampledSpectrum min(Expr<float> a, const SampledSpectrum &b) noexcept;
[[nodiscard]] SampledSpectrum min(const SampledSpectrum &a, const SampledSpectrum &b) noexcept;
[[nodiscard]] SampledSpectrum clamp(const SampledSpectrum &v, Expr<float> l, Expr<float> r) noexcept;
[[nodiscard]] SampledSpectrum clamp(const SampledSpectrum &v, const SampledSpectrum &l, Expr<float> r) noexcept;
[[nodiscard]] SampledSpectrum clamp(const SampledSpectrum &v, Expr<float> l, const SampledSpectrum &r) noexcept;
[[nodiscard]] SampledSpectrum clamp(const SampledSpectrum &v, const SampledSpectrum &l, const SampledSpectrum &r) noexcept;
[[nodiscard]] Bool any(const SampledSpectrum &v) noexcept;
[[nodiscard]] Bool all(const SampledSpectrum &v) noexcept;

[[nodiscard]] SampledSpectrum zero_if_any_nan(const SampledSpectrum &t) noexcept;

// some math functions
[[nodiscard]] SampledSpectrum saturate(const SampledSpectrum &t) noexcept;
[[nodiscard]] SampledSpectrum abs(const SampledSpectrum &t) noexcept;
[[nodiscard]] SampledSpectrum sqrt(const SampledSpectrum &t) noexcept;
[[nodiscard]] SampledSpectrum exp(const SampledSpectrum &t) noexcept;
// TODO: other math functions

using luisa::lerp;
using luisa::compute::fma;
using luisa::compute::lerp;

template<typename A, typename B, typename C>
    requires std::disjunction_v<
        std::is_same<std::remove_cvref_t<A>, SampledSpectrum>,
        std::is_same<std::remove_cvref_t<B>, SampledSpectrum>,
        std::is_same<std::remove_cvref_t<C>, SampledSpectrum>>
[[nodiscard]] auto fma(const A &a, const B &b, const C &c) noexcept {
    return a * b + c;
}

template<typename A, typename B, typename T>
    requires std::disjunction_v<
        std::is_same<std::remove_cvref_t<A>, SampledSpectrum>,
        std::is_same<std::remove_cvref_t<B>, SampledSpectrum>,
        std::is_same<std::remove_cvref_t<T>, SampledSpectrum>>
[[nodiscard]] auto lerp(const A &a, const B &b, const T &t) noexcept {
    return t * (b - a) + a;
}

class SampledWavelengths {

private:
    Local<float> _lambdas;
    Local<float> _pdfs;

public:
    explicit SampledWavelengths(uint dim) noexcept : _lambdas{dim}, _pdfs{dim} {}
    [[nodiscard]] auto lambda(Expr<uint> i) const noexcept { return _lambdas[i]; }
    [[nodiscard]] auto pdf(Expr<uint> i) const noexcept { return _pdfs[i]; }
    void set_lambda(Expr<uint> i, Expr<float> lambda) noexcept { _lambdas[i] = lambda; }
    void set_pdf(Expr<uint> i, Expr<float> pdf) noexcept { _pdfs[i] = pdf; }
    [[nodiscard]] auto dimension() const noexcept { return static_cast<uint>(_lambdas.size()); }
    void terminate_secondary() const noexcept;
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::SampledSpectrum)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::SampledWavelengths)
