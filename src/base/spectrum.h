//
// Created by Mike Smith on 2022/3/21.
//

#pragma once

#include <dsl/syntax.h>
#include <runtime/command_buffer.h>
#include <base/spd.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <util/spec.h>

namespace luisa::render {

class Pipeline;

using compute::PixelStorage;
using compute::Float4;
using compute::Expr;
using compute::Image;
using compute::ImageView;
using compute::Shader;

class Spectrum : public SceneNode {

public:
    struct Decode {
        SampledSpectrum value;
        Float strength;
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const Spectrum *_spectrum;
        SPD _cie_x;
        SPD _cie_y;
        SPD _cie_z;

    private:
        [[noreturn]] void _report_backward_unsupported_or_not_implemented() const noexcept;

    public:
        Instance(Pipeline &pipeline, CommandBuffer &cb, const Spectrum *spec) noexcept;
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }

        // clang-format off
        template<typename T = Spectrum>
            requires std::is_base_of_v<Spectrum, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_spectrum); }
        // clang-format on

        // interfaces
        [[nodiscard]] virtual SampledWavelengths sample(Expr<float> u) const noexcept;
        [[nodiscard]] virtual Float4 encode_srgb_albedo(Expr<float3> rgb) const noexcept = 0;
        [[nodiscard]] virtual Float4 encode_srgb_illuminant(Expr<float3> rgb) const noexcept = 0;
        [[nodiscard]] virtual Decode decode_albedo(const SampledWavelengths &swl, Expr<float4> v) const noexcept = 0;
        [[nodiscard]] virtual Decode decode_illuminant(const SampledWavelengths &swl, Expr<float4> v) const noexcept = 0;
        [[nodiscard]] virtual Float cie_y(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept;
        [[nodiscard]] virtual Float3 cie_xyz(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept;
        [[nodiscard]] virtual Float3 srgb(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept;
        [[nodiscard]] virtual Float4 backward_decode_albedo(
            const SampledWavelengths &swl, Expr<float4> v, const SampledSpectrum &dSpec) const noexcept;
        [[nodiscard]] virtual Float4 backward_decode_illuminant(
            const SampledWavelengths &swl, Expr<float4> v, const SampledSpectrum &dSpec) const noexcept;
        [[nodiscard]] virtual SampledSpectrum backward_cie_y(
            const SampledWavelengths &swl, const SampledSpectrum &sp, Expr<float> dY) const noexcept;
        [[nodiscard]] virtual SampledSpectrum backward_cie_xyz(
            const SampledWavelengths &swl, const SampledSpectrum &sp, Expr<float3> dXYZ) const noexcept;
        [[nodiscard]] virtual SampledSpectrum backward_srgb(
            const SampledWavelengths &swl, const SampledSpectrum &sp, Expr<float3> dSRGB) const noexcept;
    };

public:
    Spectrum(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_differentiable() const noexcept = 0;
    [[nodiscard]] virtual bool is_fixed() const noexcept = 0;
    [[nodiscard]] virtual bool requires_encoding() const noexcept = 0;
    [[nodiscard]] virtual uint dimension() const noexcept = 0;
    [[nodiscard]] virtual float4 encode_srgb_albedo(float3 rgb) const noexcept = 0;
    [[nodiscard]] virtual float4 encode_srgb_illuminant(float3 rgb) const noexcept = 0;
    [[nodiscard]] virtual PixelStorage encoded_albedo_storage(PixelStorage storage) const noexcept = 0;
    [[nodiscard]] virtual PixelStorage encoded_illuminant_storage(PixelStorage storage) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render
