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
#include <util/polymorphic_closure.h>

#include <utility>

namespace luisa::render {

using compute::BindlessArray;
using compute::Expr;
using compute::Float3;
using compute::Local;
using compute::Var;

class Shape;
class Sampler;
class Frame;
class Interaction;

template<typename BaseSurface, typename BaseInstance>
class OpacitySurfaceWrapper;

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
            return Sample{
                .eval = Evaluation::zero(spec_dim),
                .wi = make_float3(0.f, 0.f, 1.f),
                .event = Surface::event_reflect};
        }
    };

    class Closure : public PolymorphicClosure {

    private:
        const Pipeline &_pipeline;
        const SampledWavelengths &_swl;
        Float _time;

    private:
        template<typename BS, typename BSI>
        friend class OpacitySurfaceWrapper;
        [[nodiscard]] virtual Evaluation _evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept = 0;
        [[nodiscard]] virtual Sample _sample(Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u, TransportMode mode) const noexcept = 0;

    public:
        Closure(const Pipeline &pipeline,
                const SampledWavelengths &swl,
                Expr<float> time) noexcept
            : _pipeline{pipeline},
              _swl{swl}, _time{time} {}
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] auto &swl() const noexcept { return _swl; }
        [[nodiscard]] auto time() const noexcept { return _time; }

        [[nodiscard]] Evaluation evaluate(Expr<float3> wo, Expr<float3> wi,
                                          TransportMode mode = TransportMode::RADIANCE) const noexcept;
        [[nodiscard]] Sample sample(Expr<float3> wo,
                                    Expr<float> u_lobe, Expr<float2> u,
                                    TransportMode mode = TransportMode::RADIANCE) const noexcept;

        // surface properties
        [[nodiscard]] virtual luisa::optional<Float> opacity() const noexcept { return nullopt; }     // nullopt if never possible to be non-opaque
        [[nodiscard]] virtual luisa::optional<Float> eta() const noexcept { return nullopt; }         // nullopt if never possible to be transmissive
        [[nodiscard]] virtual luisa::optional<Bool> is_dispersive() const noexcept { return nullopt; }// nullopt if never possible to be dispersive
        [[nodiscard]] virtual const Interaction &it() const noexcept = 0;
        [[nodiscard]] virtual SampledSpectrum albedo() const noexcept = 0;// albedo, might not be exact, for AOV only
        [[nodiscard]] virtual Float2 roughness() const noexcept = 0;      // roughness, might not be exact, for AOV only
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const Surface *_surface;

    private:
        friend class Surface;

    public:
        [[nodiscard]] virtual luisa::string closure_identifier() const noexcept;
        virtual void populate_closure(
            Closure *closure, const Interaction &it,
            Expr<float3> wo, Expr<float> eta_i) const noexcept = 0;
        [[nodiscard]] virtual luisa::unique_ptr<Closure> create_closure(
            const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;

    public:
        Instance(const Pipeline &pipeline, const Surface *surface) noexcept
            : _pipeline{pipeline}, _surface{surface} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Surface>
            requires std::is_base_of_v<Surface, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_surface); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }

        void closure(PolymorphicCall<Closure> &call,
                     const Interaction &it, const SampledWavelengths &swl,
                     Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept;
    };

public:
    static constexpr auto property_reflective = 1u << 0u;
    static constexpr auto property_transmissive = 1u << 1u;
    static constexpr auto property_thin = 1u << 2u;

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
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept;
};

template<typename BaseSurface,
         typename BaseInstance = typename BaseSurface::Instance>
class OpacitySurfaceWrapper : public BaseSurface {

    static_assert(std::derived_from<BaseSurface, Surface> &&
                  std::derived_from<BaseInstance, Surface::Instance>);

public:
    class Closure final : public Surface::Closure {

    private:
        luisa::unique_ptr<Surface::Closure> _base;

    public:
        struct Context {
            Float opacity;
        };

    public:
        Closure(const Pipeline &pipeline,
                const SampledWavelengths &swl,
                Expr<float> time,
                luisa::unique_ptr<Surface::Closure> base) noexcept
            : Surface::Closure{pipeline, swl, time},
              _base{std::move(base)} {}
        [[nodiscard]] auto base() const noexcept { return _base.get(); }

    public:
        [[nodiscard]] SampledSpectrum albedo() const noexcept override {
            return _base->albedo();
        }
        [[nodiscard]] Float2 roughness() const noexcept override {
            return _base->roughness();
        }
        [[nodiscard]] const Interaction &it() const noexcept override {
            return _base->it();
        }
        [[nodiscard]] luisa::optional<Float> opacity() const noexcept override {
            return context<Context>().opacity;
        }
        [[nodiscard]] optional<Float> eta() const noexcept override {
            return _base->eta();
        }
        [[nodiscard]] optional<Bool> is_dispersive() const noexcept override {
            return _base->is_dispersive();
        }

    public:
        void pre_eval() noexcept override { _base->pre_eval(); }
        void post_eval() noexcept override { _base->post_eval(); }

    private:
        [[nodiscard]] Surface::Evaluation _evaluate(Expr<float3> wo,
                                                    Expr<float3> wi,
                                                    TransportMode mode) const noexcept override {
            return _base->_evaluate(wo, wi, mode);
        }
        [[nodiscard]] Surface::Sample _sample(Expr<float3> wo,
                                              Expr<float> u_lobe, Expr<float2> u,
                                              TransportMode mode) const noexcept override {
            return _base->_sample(wo, u_lobe, u, mode);
        }
    };

    class Instance : public BaseInstance {

    private:
        const Texture::Instance *_opacity;

    public:
        Instance(BaseInstance &&base, const Texture::Instance *opacity) noexcept
            : BaseInstance{std::move(base)}, _opacity{opacity} {}

    public:
        [[nodiscard]] luisa::string closure_identifier() const noexcept override {
            auto base_identifier = BaseInstance::closure_identifier();
            if (_opacity == nullptr) { return base_identifier; }
            return luisa::format("opacity<{}>", base_identifier);
        }
        [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(
            const SampledWavelengths &swl, Expr<float> time) const noexcept override {
            auto base = BaseInstance::create_closure(swl, time);
            if (_opacity == nullptr) { return base; }
            Closure cls{this->pipeline(), swl, time, std::move(base)};
            return luisa::make_unique<Closure>(std::move(cls));
        }
        void populate_closure(Surface::Closure *closure_in, const Interaction &it,
                              Expr<float3> wo, Expr<float> eta_i) const noexcept override {
            if (_opacity == nullptr) {
                BaseInstance::populate_closure(closure_in, it, wo, eta_i);
                return;
            }
            auto closure = static_cast<Closure *>(closure_in);
            auto &swl = closure->swl();
            auto time = closure->time();
            BaseInstance::populate_closure(closure->base(), it, wo, eta_i);
            auto o = _opacity->evaluate(it, swl, time).x;
            typename Closure::Context ctx{.opacity = o};
            closure->bind(std::move(ctx));
        }
    };

private:
    const Texture *_opacity;

public:
    OpacitySurfaceWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : BaseSurface{scene, desc},
          _opacity{[](auto scene, auto desc) noexcept {
              return scene->load_texture(desc->property_node_or_default(
                  "alpha", lazy_construct([desc] {
                      return desc->property_node_or_default("opacity");
                  })));
          }(scene, desc)} {}

protected:
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto base = BaseSurface::_build(pipeline, command_buffer);
        return luisa::make_unique<Instance>(
            std::move(dynamic_cast<BaseInstance &>(*base)),
            [this](auto &pipeline, auto &command_buffer) noexcept {
                return pipeline.build_texture(command_buffer, _opacity);
            }(pipeline, command_buffer));
    }
};

template<typename BaseSurface,
         typename BaseInstance = typename BaseSurface::Instance>
class NormalMapWrapper : public BaseSurface {

    static_assert(std::derived_from<BaseSurface, Surface> &&
                  std::derived_from<BaseInstance, Surface::Instance>);

public:
    class Instance : public BaseInstance {

    private:
        const Texture::Instance *_map;
        float _strength;

    public:
        Instance(BaseInstance &&base, const Texture::Instance *normal, float strength) noexcept
            : BaseInstance{std::move(base)}, _map{normal}, _strength{strength} {}

    public:
        void populate_closure(Surface::Closure *closure, const Interaction &it,
                              Expr<float3> wo, Expr<float> eta_i) const noexcept override {
            if (_map == nullptr) {
                BaseInstance::populate_closure(closure, it, wo, eta_i);
                return;
            }

            auto &swl = closure->swl();
            auto time = closure->time();

            auto normal_local = 2.f * _map->evaluate(it, swl, time).xyz() - 1.f;
            if (_strength != 1.f) { normal_local *= make_float3(_strength, _strength, 1.f); }
            auto mapped_it = it;
            auto normal = it.shading().local_to_world(normal_local);
            mapped_it.set_shading(Frame::make(clamp_shading_normal(normal, it.ng(), wo),
                                              it.shading().s()));

            BaseInstance::populate_closure(closure, mapped_it, wo, eta_i);
        }
    };

private:
    const Texture *_normal_map;
    float _strength;

public:
    NormalMapWrapper(Scene *scene, const SceneNodeDesc *desc) noexcept
        : BaseSurface{scene, desc},
          _normal_map{[](auto scene, auto desc) noexcept {
              return scene->load_texture(desc->property_node_or_default("normal_map"));
          }(scene, desc)},
          _strength{desc->property_float_or_default("normal_map_strength", 1.f)} {}

protected:
    [[nodiscard]] luisa::unique_ptr<Surface::Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto base = BaseSurface::_build(pipeline, command_buffer);
        return luisa::make_unique<Instance>(
            std::move(dynamic_cast<BaseInstance &>(*base)),
            [this](auto &pipeline, auto &command_buffer) noexcept {
                return pipeline.build_texture(command_buffer, _normal_map);
            }(pipeline, command_buffer),
            _strength);
    }
};

template<typename BaseSurface,
         typename BaseInstance = typename BaseSurface::Instance>
class TwoSidedWrapper : public BaseSurface {

    static_assert(std::derived_from<BaseSurface, Surface> &&
                  std::derived_from<BaseInstance, Surface::Instance>);

public:
    class Instance : public BaseInstance {

    private:
        bool _two_sided;

    public:
        Instance(BaseInstance &&base, bool two_sided) noexcept
            : BaseInstance{std::move(base)}, _two_sided{two_sided} {}

    public:
        void populate_closure(Surface::Closure *closure, const Interaction &it,
                              Expr<float3> wo, Expr<float> eta_i) const noexcept override {
            if (_two_sided) {
                auto it_copy = it;
                it_copy.shading().flip();
                BaseInstance::populate_closure(closure, it_copy, wo, eta_i);
            }
            BaseInstance::populate_closure(closure, it, wo, eta_i);
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
            std::move(dynamic_cast<BaseInstance &>(*base)), _two_sided);
    }
    [[nodiscard]] uint properties() const noexcept override {
        auto p = BaseSurface::properties();
        return _two_sided ? p & (~Surface::property_transmissive) : p;
    }
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Surface::Instance)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Surface::Closure)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Surface::Sample)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Surface::Evaluation)
