//
// Created by Mike on 2021/12/14.
//

#pragma once

#include <util/spec.h>
#include <util/scattering.h>
#include <base/scene_node.h>
#include <base/texture.h>
#include <base/sampler.h>
#include <base/spectrum.h>
#include <base/interaction.h>

#include <utility>

namespace luisa::render {

using compute::BindlessArray;
using compute::Expr;
using compute::Float3;
using compute::Var;

class Shape;
class Sampler;
class Frame;
class Interaction;

class Surface : public SceneNode {

public:
    static constexpr auto event_reflect = 0x00u;
    static constexpr auto event_enter = 0x01u;
    static constexpr auto event_exit = 0x02u;
    static constexpr auto event_transmit = event_enter | event_exit;
    static constexpr auto event_through = 0x04u;

    struct Evaluation {
        SampledSpectrum f;
        Float pdf;
        [[nodiscard]] static auto zero(uint spec_dim) noexcept {
            return Evaluation{
                .f = SampledSpectrum{spec_dim},
                .pdf = 0.f};
        }
    };

    struct Sample {
        Evaluation eval;
        Float3 wi;
        UInt event;
        [[nodiscard]] static auto zero(uint spec_dim) noexcept {
            return Sample{.eval = Evaluation::zero(spec_dim),
                          .wi = make_float3(0.f, 0.f, 1.f),
                          .event = event_reflect};
        }
    };

    class Instance;

    class Closure {

    private:
        const Instance *_instance;

    protected:
        Interaction _it;
        const SampledWavelengths &_swl;
        Float _time;

    private:
        [[nodiscard]] virtual Evaluation _evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept = 0;
        [[nodiscard]] virtual Sample _sample(Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u, TransportMode mode) const noexcept = 0;
        virtual void _backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df, TransportMode mode) const noexcept = 0;

    private:
        [[nodiscard]] virtual luisa::optional<Float> _opacity() const noexcept;
        [[nodiscard]] virtual luisa::optional<Bool> _is_dispersive() const noexcept;
        [[nodiscard]] virtual luisa::optional<Float> _eta() const noexcept;

    public:
        Closure(const Instance *instance, Interaction it,
                const SampledWavelengths &swl, Expr<float> time) noexcept;
        virtual ~Closure() noexcept = default;
        template<typename T = Instance>
            requires std::is_base_of_v<Instance, T>
        [[nodiscard]] auto instance() const noexcept { return static_cast<const T *>(_instance); }
        [[nodiscard]] Evaluation evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode = TransportMode::RADIANCE) const noexcept;
        [[nodiscard]] Sample sample(Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u, TransportMode mode = TransportMode::RADIANCE) const noexcept;
        void backward(Expr<float3> wo, Expr<float3> wi, const SampledSpectrum &df, TransportMode mode = TransportMode::RADIANCE) const noexcept;
        [[nodiscard]] auto &swl() const noexcept { return _swl; }          // sampled wavelengths
        [[nodiscard]] auto &it() const noexcept { return _it; }            // interaction, possibly normal mapped
        [[nodiscard]] luisa::optional<Float> opacity() const noexcept;     // nullopt if never possible to be non-opaque
        [[nodiscard]] luisa::optional<Float> eta() const noexcept;         // nullopt if never possible to be transmissive
        [[nodiscard]] luisa::optional<Bool> is_dispersive() const noexcept;// nullopt if never possible to be dispersive
        [[nodiscard]] virtual SampledSpectrum albedo() const noexcept = 0; // albedo, might not be exact, for AOV only
        [[nodiscard]] virtual Float2 roughness() const noexcept = 0;       // roughness, might not be exact, for AOV only
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const Surface *_surface;

    private:
        friend class Surface;

    public:
        Instance(const Pipeline &pipeline, const Surface *surface) noexcept
            : _pipeline{pipeline}, _surface{surface} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Surface>
            requires std::is_base_of_v<Surface, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_surface); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual luisa::unique_ptr<Closure> closure(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> eta_i, Expr<float> time) const noexcept = 0;
    };

public:
    static constexpr auto property_reflective = 1u << 0u;
    static constexpr auto property_transmissive = 1u << 1u;
    static constexpr auto property_thin = 1u << 2u;
    static constexpr auto property_differentiable = 1u << 3u;

protected:
    [[nodiscard]] virtual luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;

public:
    Surface(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual uint properties() const noexcept = 0;
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] auto is_reflective() const noexcept { return static_cast<bool>(properties() & property_reflective); }
    [[nodiscard]] auto is_transmissive() const noexcept { return static_cast<bool>(properties() & property_transmissive); }
    [[nodiscard]] auto is_thin() const noexcept { return static_cast<bool>(properties() & property_thin); }
    [[nodiscard]] auto is_differentiable() const noexcept { return static_cast<bool>(properties() & property_differentiable); }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept;
};

template<typename BaseSurface,
         typename BaseInstance = typename BaseSurface::Instance,
         typename BaseClosure = typename BaseSurface::Closure>
class OpacitySurfaceWrapper : public BaseSurface {

public:
    class Closure final : public BaseClosure {

    private:
        luisa::optional<Float> _alpha;
        [[nodiscard]] luisa::optional<Float> _opacity() const noexcept override { return _alpha; }

    public:
        [[nodiscard]] Closure(BaseClosure &&base, luisa::optional<Float> alpha) noexcept
            : BaseClosure{std::move(base)}, _alpha{std::move(alpha)} {}
    };

    class Instance : public BaseInstance {

    private:
        const Texture::Instance *_opacity;

    public:
        Instance(BaseInstance &&base, const Texture::Instance *opacity) noexcept
            : BaseInstance{std::move(base)}, _opacity{opacity} {}
        [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> eta_i, Expr<float> time) const noexcept override {
            auto base = BaseInstance::closure(it, swl, eta_i, time);
            auto alpha = _opacity ? luisa::make_optional(_opacity->evaluate(it, swl, time).x) : luisa::nullopt;
            return luisa::make_unique<Closure>(std::move(*dynamic_cast<BaseClosure *>(base.release())), std::move(alpha));
        }
    };

private:
    const Texture *_opacity;

public:
    OpacitySurfaceWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : BaseSurface{scene, desc},
          _opacity{[](auto scene, auto desc) noexcept {
              return scene->load_texture(desc->property_node_or_default("alpha"));
          }(scene, desc)} {}

protected:
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto base = BaseSurface::_build(pipeline, command_buffer);
        return luisa::make_unique<Instance>(
            std::move(*dynamic_cast<BaseInstance *>(base.release())),
            [this](auto &pipeline, auto &command_buffer) noexcept {
                return pipeline.build_texture(command_buffer, _opacity);
            }(pipeline, command_buffer));
    }
};

template<typename BaseSurface,
         typename BaseInstance = typename BaseSurface::Instance>
class NormalMapWrapper : public BaseSurface {

public:
    class Instance : public BaseInstance {

    private:
        const Texture::Instance *_map;

    public:
        Instance(BaseInstance &&base, const Texture::Instance *normal) noexcept
            : BaseInstance{std::move(base)}, _map{normal} {}
        [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> eta_i, Expr<float> time) const noexcept override {
            if (_map == nullptr) {
                return BaseInstance::closure(it, swl, eta_i, time);
            }
            auto normal_local = 2.f * _map->evaluate(it, swl, time).xyz() - 1.f;
            auto normal = it.shading().local_to_world(normal_local);
            auto mapped_it = it;
            mapped_it.set_shading(Frame::make(normal, it.shading().u()));
            return BaseInstance::closure(mapped_it, swl, eta_i, time);
        }
    };

private:
    const Texture *_normal_map;

public:
    NormalMapWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : BaseSurface{scene, desc},
          _normal_map{[](auto scene, auto desc) noexcept {
              return scene->load_texture(desc->property_node_or_default("normal_map"));
          }(scene, desc)} {}

protected:
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto base = BaseSurface::_build(pipeline, command_buffer);
        return luisa::make_unique<Instance>(
            std::move(*dynamic_cast<BaseInstance *>(base.release())),
            [this](auto &pipeline, auto &command_buffer) noexcept {
                return pipeline.build_texture(command_buffer, _normal_map);
            }(pipeline, command_buffer));
    }
};

template<typename BaseSurface,
         typename BaseInstance = typename BaseSurface::Instance>
class TwoSidedWrapper : public BaseSurface {

public:
    class Instance : public BaseInstance {

    private:
        bool _two_sided;

    public:
        Instance(BaseInstance &&base, bool two_sided) noexcept
            : BaseInstance{std::move(base)}, _two_sided{two_sided} {}
        [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
            const Interaction &it, const SampledWavelengths &swl,
            Expr<float> eta_i, Expr<float> time) const noexcept override {
            if (_two_sided) {
                Interaction it_copy{
                    it.shape(), it.instance_id(), it.triangle_id(),
                    it.triangle_area(), it.p(), it.ng(), it.uv(), it.p_shading(),
                    ite(it.back_facing(), -1.f, 1.f) * it.shading().n(),
                    it.shading().u(), false};
                return BaseInstance::closure(it_copy, swl, eta_i, time);
            }
            return BaseInstance::closure(it, swl, eta_i, time);
        }
    };

private:
    bool _two_sided;

public:
    TwoSidedWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : BaseSurface{scene, desc},
          _two_sided{desc->property_bool_or_default("two_sided", false)} {}

protected:
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto base = BaseSurface::_build(pipeline, command_buffer);
        return luisa::make_unique<Instance>(
            std::move(*dynamic_cast<BaseInstance *>(base.release())),
            _two_sided);
    }
    [[nodiscard]] uint properties() const noexcept override {
        auto p = BaseSurface::properties();
        return _two_sided ? p & (~Surface::property_transmissive) : p;
    }
};

}// namespace luisa::render
