//
// Created by Mike Smith on 2022/3/21.
//

#pragma once

#include <dsl/syntax.h>
#include <runtime/command_buffer.h>
#include <base/scene_node.h>
#include <base/sampler.h>
#include <util/spec.h>

namespace luisa::render {

class Pipeline;
class SampledWavelengths;

class Spectrum : public SceneNode {

public:
    class Instance {

    private:
        const Pipeline &_pipeline;
        const Spectrum *_spectrum;

    public:
        Instance(const Pipeline &pipeline, const Spectrum *spec) noexcept
            : _pipeline{pipeline}, _spectrum{spec} {}
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        template<typename T = Spectrum>
            requires std::is_base_of_v<Spectrum, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_spectrum); }

        // interfaces
        [[nodiscard]] virtual SampledWavelengths sample(Sampler::Instance &sampler) noexcept;
        [[nodiscard]] virtual SampledSpectrum albedo_from_srgb(const SampledWavelengths &swl, Expr<float3> rgb) const noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum illuminant_from_srgb(const SampledWavelengths &swl, Expr<float3> rgb) const noexcept = 0;
        [[nodiscard]] virtual Float cie_y(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept;
        [[nodiscard]] virtual Float3 cie_xyz(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept;
        [[nodiscard]] virtual Float3 srgb(const SampledWavelengths &swl, const SampledSpectrum &sp) const noexcept;
        [[nodiscard]] virtual Float3 backward_albedo_from_srgb(
            const SampledWavelengths &swl, Expr<float3> rgb,
            const SampledSpectrum &dSpec) const noexcept = 0;
        [[nodiscard]] virtual Float3 backward_illuminant_from_srgb(
            const SampledWavelengths &swl, Expr<float3> rgb,
            const SampledSpectrum &dSpec) const noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum backward_cie_y(
            const SampledWavelengths &swl, const SampledSpectrum &sp,
            Expr<float> dY) const noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum backward_cie_xyz(
            const SampledWavelengths &swl, const SampledSpectrum &sp,
            Expr<float3> dXYZ) const noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum backward_srgb(
            const SampledWavelengths &swl, const SampledSpectrum &sp,
            Expr<float3> dSRGB) const noexcept = 0;
    };

public:
    Spectrum(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_differentiable() const noexcept = 0;
    [[nodiscard]] virtual bool wavelengths_fixed() const noexcept = 0;
    [[nodiscard]] virtual uint dimension() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

class SampledWavelengths {

private:
    const Spectrum::Instance *_spectrum;
    Local<float> _lambdas;
    Local<float> _pdfs;

public:
    explicit SampledWavelengths(const Spectrum::Instance *spec) noexcept;
    [[nodiscard]] auto lambda(Expr<uint> i) const noexcept { return _lambdas[i]; }
    [[nodiscard]] auto pdf(Expr<uint> i) const noexcept { return _pdfs[i]; }
    void set_lambda(Expr<uint> i, Expr<float> lambda) noexcept { _lambdas[i] = lambda; }
    void set_pdf(Expr<uint> i, Expr<float> pdf) noexcept { _pdfs[i] = pdf; }
    [[nodiscard]] auto spectrum() const noexcept { return _spectrum; }
    [[nodiscard]] auto dimension() const noexcept { return _lambdas.size(); }
    [[nodiscard]] SampledSpectrum albedo_from_srgb(Expr<float3> rgb) const noexcept;
    [[nodiscard]] SampledSpectrum illuminant_from_srgb(Expr<float3> rgb) const noexcept;
    [[nodiscard]] Float cie_y(const SampledSpectrum &s) const noexcept;
    [[nodiscard]] Float3 cie_xyz(const SampledSpectrum &s) const noexcept;
    [[nodiscard]] Float3 srgb(const SampledSpectrum &s) const noexcept;
    [[nodiscard]] Float3 backward_albedo_from_srgb(Expr<float3> rgb, const SampledSpectrum &dSpec) const noexcept;
    [[nodiscard]] Float3 backward_illuminant_from_srgb(Expr<float3> rgb, const SampledSpectrum &dSpec) const noexcept;
    [[nodiscard]] SampledSpectrum backward_cie_y(const SampledSpectrum &s, Expr<float> dY) const noexcept;
    [[nodiscard]] SampledSpectrum backward_cie_xyz(const SampledSpectrum &s, Expr<float3> dXYZ) const noexcept;
    [[nodiscard]] SampledSpectrum backward_srgb(const SampledSpectrum &s, Expr<float3> dSRGB) const noexcept;
};

}// namespace luisa::render
