//
// Created by Mike Smith on 2022/1/19.
//

#pragma once

#include <core/basic_types.h>
#include <core/stl.h>
#include <dsl/syntax.h>
#include "colorspace.h"

namespace luisa::render {

constexpr auto visible_wavelength_min = 360.0f;
constexpr auto visible_wavelength_max = 830.0f;

using compute::Expr;
using compute::Bool;
using compute::Float;
using compute::Float4;

class RGBSigmoidPolynomial {

private:
    Float _c0;
    Float _c1;
    Float _c2;

private:
    [[nodiscard]] static Float _s(Expr<float> x) noexcept;

public:
    RGBSigmoidPolynomial() noexcept = default;
    RGBSigmoidPolynomial(Expr<float> c0, Expr<float> c1, Expr<float> c2) noexcept
        : _c0{c0}, _c1{c1}, _c2{c2} {}
    [[nodiscard]] Float operator()(Expr<float> lambda) const noexcept;
    [[nodiscard]] Float maximum() const noexcept;
};

class Spectrum {

public:
    virtual ~Spectrum() noexcept = default;
};

class SampledWavelengths {

public:
    static constexpr auto sample_count = 4u;

private:
    Float4 _lambda;
    Float4 _pdf;

public:
    SampledWavelengths() noexcept = default;
    SampledWavelengths(Expr<float4> lambda, Expr<float4> pdf) noexcept
        : _lambda{lambda}, _pdf{pdf} {}
    [[nodiscard]] Bool operator==(const SampledWavelengths &rhs) const noexcept;
    [[nodiscard]] Bool operator!=(const SampledWavelengths &rhs) const noexcept;
    [[nodiscard]] Bool secondary_terminated() const noexcept;
    void terminate_secondary() noexcept;
    [[nodiscard]] static SampledWavelengths sample_uniform(
        Expr<float> u,
        Expr<float> lambda_min = visible_wavelength_min,
        Expr<float> lambda_max = visible_wavelength_max) noexcept;
    [[nodiscard]] static SampledWavelengths sample_visible(Expr<float> u) noexcept;
    [[nodiscard]] auto lambda() const noexcept { return _lambda; }
    [[nodiscard]] auto pdf() const noexcept { return _pdf; }
};

[[nodiscard]] Float sampled_spectrum_to_y(const SampledWavelengths &lambda, Expr<float4> values) noexcept;
[[nodiscard]] Float3 sampled_spectrum_to_xyz(const SampledWavelengths &lambda, Expr<float4> values) noexcept;
[[nodiscard]] Float3 sampled_spectrum_to_rgb(const SampledWavelengths &lambda, Expr<float4> values) noexcept;

}
