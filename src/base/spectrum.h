//
// Created by Mike Smith on 2022/3/21.
//

#pragma once

#include <dsl/syntax.h>
#include <util/command_buffer.h>
#include <base/spd.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <util/spec.h>

namespace luisa::render {

class Pipeline;

using compute::Expr;
using compute::Float4;
using compute::Image;
using compute::ImageView;
using compute::PixelStorage;
using compute::Shader;

class Spectrum : public SceneNode {

public:
    struct Decode {
        SampledSpectrum value;
        Float strength;
        [[nodiscard]] static auto constant(uint dim, float value) noexcept {
            return Decode{.value = {dim, value}, .strength = value};
        }
        [[nodiscard]] static auto one(uint dim) noexcept { return constant(dim, 1.f); }
        [[nodiscard]] static auto zero(uint dim) noexcept { return constant(dim, 0.f); }
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const Spectrum *_spectrum;
        SPD _cie_x;
        SPD _cie_y;
        SPD _cie_z;

    public:
        Instance(Pipeline &pipeline, CommandBuffer &cb, const Spectrum *spec) noexcept;
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }

        template<typename T = Spectrum>
            requires std::is_base_of_v<Spectrum, T>
        [[nodiscard]] auto node() const noexcept {
            return static_cast<const T *>(_spectrum);
        }

        // interfaces
        [[nodiscard]] virtual SampledWavelengths sample(Expr<float> u) const noexcept;
        [[nodiscard]] virtual Float4 encode_srgb_albedo(Expr<float3> rgb) const noexcept = 0;
        [[nodiscard]] virtual Float4 encode_srgb_unbounded(Expr<float3> rgb) const noexcept = 0;
        [[nodiscard]] virtual Float4 encode_srgb_illuminant(Expr<float3> rgb) const noexcept = 0;
        [[nodiscard]] virtual Decode decode_albedo(const SampledWavelengths &swl, Expr<float4> v) const noexcept = 0;
        [[nodiscard]] virtual Decode decode_unbounded(const SampledWavelengths &swl, Expr<float4> v) const noexcept = 0;
        [[nodiscard]] virtual Decode decode_illuminant(const SampledWavelengths &swl, Expr<float4> v) const noexcept = 0;
        [[nodiscard]] virtual Float cie_y(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept;
        [[nodiscard]] virtual Float3 cie_xyz(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept;
        [[nodiscard]] virtual Float3 srgb(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept;
        //get target_sp*sp, in the wavelengths of target_swl
        [[nodiscard]] virtual Float3 wavelength_mul(const SampledWavelengths &target_swl, const SampledSpectrum &target_sp,
                                                    const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept;
    };

public:
    Spectrum(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_fixed() const noexcept = 0;
    [[nodiscard]] virtual uint dimension() const noexcept = 0;
    [[nodiscard]] virtual float4 encode_static_srgb_albedo(float3 rgb) const noexcept = 0;
    [[nodiscard]] virtual float4 encode_static_srgb_unbounded(float3 rgb) const noexcept = 0;
    [[nodiscard]] virtual float4 encode_static_srgb_illuminant(float3 rgb) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Spectrum::Instance)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Spectrum::Decode)
