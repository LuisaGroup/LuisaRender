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

using compute::BindlessArray;
using compute::Bool;
using compute::CommandBuffer;
using compute::Constant;
using compute::Expr;
using compute::Float;
using compute::Float4;
using compute::VolumeView;

class RGBSigmoidPolynomial {

private:
    Float _c0;
    Float _c1;
    Float _c2;
    luisa::optional<Float> _maximum;

private:
    [[nodiscard]] static Float _s(Expr<float> x) noexcept;
    [[nodiscard]] static Float4 _s(Expr<float4> x) noexcept;

public:
    RGBSigmoidPolynomial() noexcept = default;
    RGBSigmoidPolynomial(Expr<float> c0, Expr<float> c1, Expr<float> c2) noexcept
        : _c0{c0}, _c1{c1}, _c2{c2} {}
    explicit RGBSigmoidPolynomial(Expr<float3> c) noexcept
        : RGBSigmoidPolynomial{c.x, c.y, c.z} {}
    [[nodiscard]] Float operator()(Expr<float> lambda) const noexcept;
    [[nodiscard]] Float4 operator()(Expr<float4> lambda) const noexcept;
    [[nodiscard]] Float maximum() const noexcept;
};

class RGB2SpectrumTable {

public:
    static constexpr auto resolution = 64u;

private:
    using scale_table_type = const float[resolution];
    using coefficient_table_type = const float[3][resolution][resolution][resolution][3];
    const scale_table_type &_z_nodes;
    const coefficient_table_type &_coefficients;

public:
    constexpr RGB2SpectrumTable(const scale_table_type &z_nodes, const coefficient_table_type &coefficients) noexcept
        : _z_nodes{z_nodes}, _coefficients{coefficients} {}
    constexpr RGB2SpectrumTable(RGB2SpectrumTable &&) noexcept = default;
    constexpr RGB2SpectrumTable(const RGB2SpectrumTable &) noexcept = default;
    [[nodiscard]] static RGB2SpectrumTable srgb() noexcept;
    [[nodiscard]] float3 decode_albedo(float3 rgb) const noexcept;
    [[nodiscard]] std::pair<float3, float> decode_unbound(float3 rgb) const noexcept;
    [[nodiscard]] RGBSigmoidPolynomial decode_albedo(Expr<BindlessArray> array, Expr<uint> base_index, Expr<float3> rgb) const noexcept;
    [[nodiscard]] std::pair<RGBSigmoidPolynomial, Float> decode_unbound(Expr<BindlessArray> array, Expr<uint> base_index, Expr<float3> rgb) const noexcept;
    void encode(CommandBuffer &command_buffer, VolumeView<float> t0, VolumeView<float> t1, VolumeView<float> t2) const noexcept;
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
    [[nodiscard]] Float cie_y(Expr<float4> values) const noexcept;
    [[nodiscard]] Float3 cie_xyz(Expr<float4> values) const noexcept;
    [[nodiscard]] Float3 srgb(Expr<float4> values) const noexcept;
};

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
    [[nodiscard]] static const DenselySampledSpectrum &cie_illum_d6500() noexcept;
    [[nodiscard]] Float4 sample(const SampledWavelengths &swl) const noexcept;
};

class RGBAlbedoSpectrum {

private:
    RGBSigmoidPolynomial _rsp;

public:
    explicit RGBAlbedoSpectrum(RGBSigmoidPolynomial rsp) noexcept
        : _rsp{std::move(rsp)} {}
    [[nodiscard]] auto sample(const SampledWavelengths &swl) const noexcept {
        return _rsp(swl.lambda());
    }
};

class RGBUnboundSpectrum {

private:
    RGBSigmoidPolynomial _rsp;
    Float _scale;

public:
    RGBUnboundSpectrum(RGBSigmoidPolynomial rsp, Expr<float> scale) noexcept
        : _rsp{std::move(rsp)}, _scale{scale} {}
    [[nodiscard]] auto sample(const SampledWavelengths &swl) const noexcept {
        return _rsp(swl.lambda()) * _scale;
    }
};

class RGBIlluminantSpectrum {

private:
    RGBSigmoidPolynomial _rsp;
    Float _scale;
    const DenselySampledSpectrum &_illuminant;

public:
    RGBIlluminantSpectrum(RGBSigmoidPolynomial rsp, Expr<float> scale, const DenselySampledSpectrum &illum) noexcept
        : _rsp{std::move(rsp)}, _scale{scale}, _illuminant{illum} {}
    [[nodiscard]] auto sample(const SampledWavelengths &swl) const noexcept {
        return _rsp(swl.lambda()) * _scale * _illuminant.sample(swl);
    }
};

}// namespace luisa::render
