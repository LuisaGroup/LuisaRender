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

using compute::ArrayVar;
using compute::CommandBuffer;
using compute::Expr;
using compute::Float;
using compute::Float3;

class SampledSpectrum;

class Spectrum : public SceneNode {

public:
    enum struct Category {
        ALBEDO,
        UNBOUND,
        ILLUMINANT
    };

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
        virtual void sample_wavelengths(const Sampler::Instance &sampler) noexcept = 0;
        virtual void load_wavelengths(Expr<uint> state_id) noexcept = 0;
        virtual void save_wavelengths(Expr<uint> state_id) noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum allocate() const noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum from_srgb(Expr<float3> rgb, Category category) const noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum from_dense(const DenselySampledSpectrum &dense) const noexcept = 0;
        [[nodiscard]] virtual Float cie_y(const SampledSpectrum &sp) const noexcept = 0;
        [[nodiscard]] virtual Float3 cie_xyz(const SampledSpectrum &sp) const noexcept = 0;
        [[nodiscard]] virtual Float3 srgb(const SampledSpectrum &sp) const noexcept;
    };

public:
    Spectrum(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_continuous() const noexcept = 0;
    [[nodiscard]] virtual bool is_differentiable() const noexcept = 0;
    [[nodiscard]] virtual uint dimension() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;
};

struct SampledSpectrumBase {
    SampledSpectrumBase() noexcept = default;
    SampledSpectrumBase(SampledSpectrumBase &&) noexcept = delete;
    SampledSpectrumBase(const SampledSpectrumBase &) noexcept = delete;
    SampledSpectrumBase &operator=(SampledSpectrumBase &&) noexcept = delete;
    SampledSpectrumBase &operator=(const SampledSpectrumBase &) noexcept = delete;
    virtual ~SampledSpectrumBase() noexcept = default;
    [[nodiscard]] virtual Float &at(Expr<uint> index) noexcept = 0;
    [[nodiscard]] virtual Float at(Expr<uint> index) const noexcept = 0;
};

template<size_t N>
class SizedSampledSpectrum final : public SampledSpectrumBase {

private:
    ArrayVar<float, N> _samples;

public:
    [[nodiscard]] Float &at(Expr<uint> index) noexcept override { return _samples[index]; }
    [[nodiscard]] Float at(Expr<uint> index) const noexcept override { return _samples[index]; }
};

class SampledSpectrum {

private:
    const Spectrum::Instance *_spectrum;
    luisa::unique_ptr<SampledSpectrumBase> _sample;

public:
    SampledSpectrum(const Spectrum::Instance *spec,
                    luisa::unique_ptr<SampledSpectrumBase> sample) noexcept
        : _spectrum{spec}, _sample{std::move(sample)} {}
    [[nodiscard]] auto dimension() const noexcept { return _spectrum->node()->dimension(); }
    [[nodiscard]] Float &operator[](Expr<uint> i) noexcept { return _sample->at(i); }
    [[nodiscard]] Float operator[](Expr<uint> i) const noexcept { return _sample->at(i); }
    template<typename F>
    void for_each(F &&f) noexcept {
        for (auto i = 0u; i < dimension(); i++) { f(i, (*this)[i]); }
    }
    template<typename F>
    void for_each(F &&f) const noexcept {
        for (auto i = 0u; i < dimension(); i++) { f(i, (*this)[i]); }
    }
    template<typename F>
    [[nodiscard]] auto map(F &&f) const noexcept {
        auto s = _spectrum->allocate();
        for (auto i = 0u; i < dimension(); i++) { s[i] = f(i, (*this)[i]); }
        return s;
    }
    template<typename T, typename F>
    [[nodiscard]] auto reduce(T &&initial, F &&f) const noexcept {
        using compute::def;
        auto r = def(std::forward<T>(initial));
        for (auto i = 0u; i < dimension(); i++) { r = f(r, i, (*this)[i]); }
        return r;
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
    }
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(+)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(-)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(*)
    LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP(/)
#undef LUISA_RENDER_SAMPLED_SPECTRUM_MAKE_BINARY_OP
};

}// namespace luisa::render
