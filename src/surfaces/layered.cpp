#include <base/surface.h>
#include <base/scene.h>
#include <base/pipeline.h>
#include <util/rng.h>
#include <util/sampling.h>

namespace luisa::render {

// The following code is from PBRT-v4.
// License: Apache 2.0
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.

class HGPhaseFunction {
public:
    struct PhaseFunctionSample {
        Float p;
        Float3 wi;
        Float pdf;
    };

    // HGPhaseFunction Public Methods
    HGPhaseFunction() noexcept = default;

    explicit HGPhaseFunction(Float g) noexcept : g(g) {}

    [[nodiscard]] Float HenyeyGreenstein(Float cosTheta, Float g) const noexcept {
        Float denom = 1 + sqr(g) + 2 * g * cosTheta;
        return inv_pi / 4.0f * (1 - sqr(g)) / (denom * sqrt(denom));
    }

    [[nodiscard]] Float3 SampleHenyeyGreenstein(Float3 wo, Float g, Float2 u, Float &pdf) const noexcept {
        // Compute $\cos \theta$ for Henyey--Greenstein sample
        Float cosTheta;
        cosTheta = ite(abs(g) < 1e-3f, 1 - 2 * u.x,
                       -1 / (2 * g) * (1 + sqr(g) - sqr((1 - sqr(g)) / (1 + g - 2 * g * u.x))));

        // Compute direction _wi_ for Henyey--Greenstein sample
        Float sinTheta = sqrt(1 - sqr(cosTheta));
        Float phi = 2 * pi * u.y;
        Frame wFrame = Frame::make(wo);
        Float3 wi = wFrame.local_to_world(spherical_direction(sinTheta, cosTheta, phi));

        pdf = HenyeyGreenstein(cosTheta, g);
        return wi;
    }

    [[nodiscard]] Float p(Float3 wo, Float3 wi) const noexcept { return HenyeyGreenstein(dot(wo, wi), g); }

    [[nodiscard]] auto Sample_p(Float3 wo, Float2 u) const noexcept {
        Float pdf;
        Float3 wi = SampleHenyeyGreenstein(wo, g, u, pdf);
        return PhaseFunctionSample{pdf, wi, pdf};
    }

    [[nodiscard]] Float PDF(Float3 wo, Float3 wi) const noexcept { return p(wo, wi); }

private:
    // HGPhaseFunction Private Members
    Float g;
};

class TopOrBottom {

private:
    const Surface::Function *_function_top;
    const Surface::Function *_function_bottom;
    const Surface::FunctionContext *_ctx_wrapper_top;
    const Surface::FunctionContext *_ctx_wrapper_bottom;
    const SampledWavelengths &_swl;
    Expr<float> _time;
    Bool _is_top;

public:
    TopOrBottom(const Surface::Function *function_top, const Surface::FunctionContext *ctx_wrapper_top,
                const Surface::Function *function_bottom, const Surface::FunctionContext *ctx_wrapper_bottom,
                const SampledWavelengths &swl, Expr<float> time,
                Expr<bool> is_top) noexcept
        : _function_top{function_top}, _function_bottom{function_bottom},
          _ctx_wrapper_top{ctx_wrapper_top}, _ctx_wrapper_bottom{ctx_wrapper_bottom},
          _swl{swl}, _time{time}, _is_top{is_top} {}

    [[nodiscard]] auto evaluate(Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept {
        auto eval = Surface::Evaluation::zero(_swl.dimension());
        $if(_is_top) {
            eval = _function_top->evaluate(_ctx_wrapper_top, _swl, _time, wo, wi, mode);
        }
        $else {
            eval = _function_bottom->evaluate(_ctx_wrapper_bottom, _swl, _time, wo, wi, mode);
        };
        return eval;
    }

    [[nodiscard]] auto sample(Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u, TransportMode mode) const noexcept {
        auto s = Surface::Sample::zero(_swl.dimension());
        $if(_is_top) {
            s = _function_top->sample(_ctx_wrapper_top, _swl, _time, wo, u_lobe, u, mode);
        }
        $else {
            s = _function_bottom->sample(_ctx_wrapper_bottom, _swl, _time, wo, u_lobe, u, mode);
        };
        return s;
    }

    [[nodiscard]] auto to_local(Expr<float3> w) const noexcept {
        return ite(_is_top,
                   _function_top->it(_ctx_wrapper_top, _time)->shading().world_to_local(w),
                   _function_bottom->it(_ctx_wrapper_bottom, _time)->shading().world_to_local(w));
    }

    [[nodiscard]] auto to_world(Expr<float3> w) const noexcept {
        return ite(_is_top,
                   _function_top->it(_ctx_wrapper_top, _time)->shading().local_to_world(w),
                   _function_bottom->it(_ctx_wrapper_bottom, _time)->shading().local_to_world(w));
    }
};

class LayeredSurface : public Surface {

private:
    const Surface *_top;
    const Surface *_bottom;
    const Texture *_thickness;
    const Texture *_g;
    const Texture *_albedo;
    uint _max_depth;
    uint _samples;

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

public:
    LayeredSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _top{scene->load_surface(desc->property_node("top"))},
          _bottom{scene->load_surface(desc->property_node("bottom"))},
          _thickness{scene->load_texture(desc->property_node_or_default("thickness"))},
          _g{scene->load_texture(desc->property_node_or_default("g"))},
          _albedo{scene->load_texture(desc->property_node_or_default("albedo"))},
          _max_depth{desc->property_uint_or_default("max_depth", 10u)},
          _samples{desc->property_uint_or_default("samples", 1u)} {
        LUISA_ASSERT(_top != nullptr && !_top->is_null() &&
                         _bottom != nullptr && !_bottom->is_null(),
                     "Creating closure for null LayeredSurface.");
    }
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto samples() const noexcept { return _samples; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::string closure_identifier() const noexcept override { return luisa::string(impl_type()); }
    [[nodiscard]] uint properties() const noexcept override {
        auto p = 0u;
        if (_top->is_thin() && _bottom->is_thin()) { p |= property_thin; }
        if (_top->is_reflective() || _bottom->is_reflective()) { p |= property_reflective; }
        if (_top->is_transmissive() && _bottom->is_transmissive()) { p |= property_transmissive; }
        return p;
    }
};

class LayeredSurfaceInstance : public Surface::Instance {

public:
    struct LayeredSurfaceContext {
        Interaction it;
        Float thickness;
        Float g;
        SampledSpectrum albedo;
        UInt max_depth;
        UInt samples;
    };

private:
    luisa::unique_ptr<Surface::Instance> _top;
    luisa::unique_ptr<Surface::Instance> _bottom;
    const Texture::Instance *_thickness;
    const Texture::Instance *_g;
    const Texture::Instance *_albedo;

public:
    LayeredSurfaceInstance(
        const Pipeline &pipeline, const LayeredSurface *surface, const Texture::Instance *thickness, const Texture::Instance *g,
        const Texture::Instance *albedo, luisa::unique_ptr<Surface::Instance> top, luisa::unique_ptr<Surface::Instance> bottom) noexcept
        : Surface::Instance{pipeline, surface},
          _top{std::move(top)}, _bottom{std::move(bottom)}, _thickness{thickness}, _g{g}, _albedo{albedo} {}
    [[nodiscard]] auto thickness() const noexcept { return _thickness; }
    [[nodiscard]] auto g() const noexcept { return _g; }
    [[nodiscard]] auto albedo() const noexcept { return _albedo; }

public:
    [[nodiscard]] uint make_closure(
        PolymorphicClosure<Surface::Function> &closure,
        luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
        Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> LayeredSurface::_build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto top = _top->build(pipeline, command_buffer);
    auto bottom = _bottom->build(pipeline, command_buffer);
    auto thickness = pipeline.build_texture(command_buffer, _thickness);
    auto g = pipeline.build_texture(command_buffer, _g);
    auto albedo = pipeline.build_texture(command_buffer, _albedo);
    return luisa::make_unique<LayeredSurfaceInstance>(
        pipeline, this, thickness, g, albedo, std::move(top), std::move(bottom));
}

class LayeredSurfaceFunction : public Surface::Function {

private:
    [[nodiscard]] static inline auto Tr(Expr<float> dz, Expr<float3> w) noexcept {
        return ite(abs(dz) <= std::numeric_limits<float>::min(),
                   1.f, exp(-abs(dz / w.z)));
    }

public:
    // TODO: are these OK?
    [[nodiscard]] SampledSpectrum albedo(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto function_top = ctx_wrapper->nested("top").function(0u);
        auto ctx_wrapper_top = ctx_wrapper->nested("top").context(0u);
        return function_top->albedo(ctx_wrapper_top, swl, time);
    }
    [[nodiscard]] Float2 roughness(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto function_top = ctx_wrapper->nested("top").function(0u);
        auto ctx_wrapper_top = ctx_wrapper->nested("top").context(0u);
        return function_top->roughness(ctx_wrapper_top, swl, time);
    }
    [[nodiscard]] const Interaction *it(const Surface::FunctionContext *ctx_wrapper, Expr<float> time) const noexcept override {
        auto ctx = ctx_wrapper->data<LayeredSurfaceInstance::LayeredSurfaceContext>();
        return &ctx->it;
    }
    [[nodiscard]] luisa::optional<Bool> is_dispersive(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto function_top = ctx_wrapper->nested("top").function(0u);
        auto ctx_wrapper_top = ctx_wrapper->nested("top").context(0u);
        auto function_bottom = ctx_wrapper->nested("bottom").function(0u);
        auto ctx_wrapper_bottom = ctx_wrapper->nested("bottom").context(0u);

        auto top_dispersive = function_top->is_dispersive(ctx_wrapper_top, swl, time);
        auto bottom_dispersive = function_bottom->is_dispersive(ctx_wrapper_bottom, swl, time);
        if (!top_dispersive) { return bottom_dispersive; }
        if (!bottom_dispersive) { return top_dispersive; }
        return *top_dispersive | *bottom_dispersive;
    }
    [[nodiscard]] luisa::optional<Float> eta(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto function_bottom = ctx_wrapper->nested("bottom").function(0u);
        auto ctx_wrapper_bottom = ctx_wrapper->nested("bottom").context(0u);

        return function_bottom->eta(ctx_wrapper_bottom, swl, time);
    }
    [[nodiscard]] luisa::optional<Float> opacity(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto function_top = ctx_wrapper->nested("top").function(0u);
        auto ctx_wrapper_top = ctx_wrapper->nested("top").context(0u);
        auto function_bottom = ctx_wrapper->nested("bottom").function(0u);
        auto ctx_wrapper_bottom = ctx_wrapper->nested("bottom").context(0u);

        auto top_opacity = function_top->opacity(ctx_wrapper_top, swl, time);
        auto bottom_opacity = function_bottom->opacity(ctx_wrapper_bottom, swl, time);
        return 1.f - ((1.f - top_opacity.value_or(1.f)) *
                      (1.f - bottom_opacity.value_or(1.f)));
    }

    [[nodiscard]] Surface::Evaluation evaluate(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time,
        Expr<float3> wo, Expr<float3> wi, TransportMode mode) const noexcept override {
        auto ctx = ctx_wrapper->data<LayeredSurfaceInstance::LayeredSurfaceContext>();
        auto &it = ctx->it;

        auto function_top = ctx_wrapper->nested("top").function(0u);
        auto ctx_wrapper_top = ctx_wrapper->nested("top").context(0u);
        auto function_bottom = ctx_wrapper->nested("bottom").function(0u);
        auto ctx_wrapper_bottom = ctx_wrapper->nested("bottom").context(0u);

        auto wi_local = it.shading().world_to_local(wi);
        auto wo_local = it.shading().world_to_local(wo);
        auto entered_top = wo_local.z > 0.f;
        auto enter_interface = TopOrBottom(function_top, ctx_wrapper_top, function_bottom, ctx_wrapper_bottom, swl, time, entered_top);
        auto exit_interface = TopOrBottom(function_bottom, ctx_wrapper_bottom, function_top, ctx_wrapper_top, swl, time, same_hemisphere(wo_local, wi_local) ^ entered_top);
        auto nonexit_interface = TopOrBottom(function_top, ctx_wrapper_top, function_bottom, ctx_wrapper_bottom, swl, time, same_hemisphere(wo_local, wi_local) ^ entered_top);
        auto exitZ = ite(same_hemisphere(wo_local, wi_local) ^ entered_top, 0.f, ctx->thickness);
        auto f = ite(same_hemisphere(wi_local, wo_local),
                     Float{ctx->samples} * enter_interface.evaluate(wo, wi, mode).f,
                     SampledSpectrum(0.f));
        auto seed = xxhash32(make_uint4(as<UInt3>(it.p()), xxhash32(as<UInt3>(wi))));
        auto pdf_sum = ite(same_hemisphere(wi_local, wo_local),
                           ite(entered_top,
                               Float{ctx->samples} * function_top->evaluate(ctx_wrapper_top, swl, time, wo, wi, mode).pdf,
                               Float{ctx->samples} * function_bottom->evaluate(ctx_wrapper_bottom, swl, time, wo, wi, mode).pdf),
                           0.f);
        // f
        $for(i, ctx->samples) {
            auto wos = enter_interface.sample(wo, lcg(seed), make_float2(lcg(seed), lcg(seed)), mode);
            $if(wos.eval.f.is_zero() | wos.eval.pdf <= 0.f) { $continue; };
            auto reverse_mode = mode == TransportMode::IMPORTANCE ? TransportMode::RADIANCE : TransportMode::IMPORTANCE;
            auto wis = exit_interface.sample(wi, lcg(seed), make_float2(lcg(seed), lcg(seed)), reverse_mode);
            auto wis_wi_local = exit_interface.to_local(wis.wi);
            $if(wis.eval.f.is_zero() | wis.eval.pdf <= 0.f) { $continue; };
            auto beta = wos.eval.f / wos.eval.pdf;
            auto z = ite(entered_top, ctx->thickness, 0.f);
            auto w = wos.wi;
            auto w_local = enter_interface.to_local(w);
            HGPhaseFunction phase(ctx->g);
            $for(depth, ctx->max_depth) {
                $if(depth > 3 & beta.max() < 0.25f) {
                    auto q = max(0.f, 1.f - beta.max());
                    $if(lcg(seed) < q) {
                        $break;
                    };
                    beta /= 1 - q;
                };
                $if(ctx->albedo.is_zero()) {
                    z = ite(z == ctx->thickness, 0.f, ctx->thickness);
                    beta *= Tr(ctx->thickness, w_local);
                }
                $else {
                    auto sigma_t = 1.f;
                    auto dz = -log(1.f - lcg(seed)) / (sigma_t / abs(w_local.z));
                    auto zp = ite(w_local.z > 0.f, z + dz, z - dz);
                    $if(z == zp) { $continue; };
                    $if(zp > 0.f & zp < ctx->thickness) {
                        auto wt = power_heuristic(1u, wis.eval.pdf, 1u,
                                                  nonexit_interface.evaluate(-w, -wis.wi, mode).pdf);
                        f += beta * ctx->albedo * phase.p(-w_local, -wis_wi_local) * wt *
                             Tr(zp - exitZ, wis_wi_local) * wis.eval.f / wis.eval.pdf;
                        auto u = make_float2(lcg(seed), lcg(seed));
                        auto ps = phase.Sample_p(-w_local, u);
                        $if(ps.pdf <= 0.f | ps.wi.z == 0.f) { $continue; };
                        beta *= ctx->albedo * ps.p / ps.pdf;
                        w_local = ps.wi;
                        w = exit_interface.to_world(w_local);
                        z = zp;
                        $if((((z < exitZ) & (w_local.z) > 0.f) | (z > exitZ & w_local.z < 0.f))) {
                            // Account for scattering through _exitInterface_
                            auto fExit = exit_interface.evaluate(-w, wi, mode).f;
                            $if(!fExit.is_zero()) {
                                Float exitPDF = exit_interface.evaluate(-w, wi, mode).pdf;
                                Float wt = power_heuristic(1u, ps.pdf, 1u, exitPDF);
                                f += beta * Tr(zp - exitZ, w_local) * fExit * wt;
                            };
                        };
                        $continue;
                    };
                    z = clamp(zp, 0.f, ctx->thickness);
                };
                $if(z == exitZ) {
                    auto uc = lcg(seed);
                    auto bs = exit_interface.sample(-w, uc, make_float2(lcg(seed), lcg(seed)), mode);
                    $if(bs.eval.f.is_zero() | bs.eval.pdf <= 0.f) { $break; };
                    beta *= bs.eval.f / bs.eval.pdf;
                    w = bs.wi;
                    w_local = exit_interface.to_local(w);
                }
                $else {
                    auto wns = nonexit_interface.evaluate(-w, -wis.wi, mode);
                    auto wt = power_heuristic(1u, wis.eval.pdf, 1u, wns.pdf);
                    f += beta * wns.f * wt * Tr(ctx->thickness, wis_wi_local) * wis.eval.f /
                         wis.eval.pdf;
                    auto uc = lcg(seed);
                    auto u = make_float2(lcg(seed), lcg(seed));
                    auto bs = nonexit_interface.sample(-w, uc, u, mode);
                    $if(bs.eval.f.is_zero() | bs.eval.pdf <= 0.f) { $break; };
                    beta *= bs.eval.f / bs.eval.pdf;
                    w = bs.wi;
                    w_local = nonexit_interface.to_local(w);
                    auto wes = exit_interface.evaluate(-w, wi, mode);
                    auto fExit = wes.f;
                    $if(!fExit.is_zero()) {
                        auto exit_pdf = wes.pdf;
                        auto wt = power_heuristic(1u, bs.eval.pdf, 1u, exit_pdf);
                        f += beta * Tr(ctx->thickness, nonexit_interface.to_local(bs.wi)) * fExit * wt;
                    };
                };
            };
        };

        // pdf
        $for(s, ctx->samples) {
            $if(same_hemisphere(wo_local, wi_local)) {
                auto r_interface = TopOrBottom(function_bottom, ctx_wrapper_bottom, function_top, ctx_wrapper_top, swl, time, entered_top);
                auto t_interface = TopOrBottom(function_top, ctx_wrapper_top, function_bottom, ctx_wrapper_bottom, swl, time, entered_top);
                auto reverse_mode = mode == TransportMode::IMPORTANCE ? TransportMode::RADIANCE : TransportMode::IMPORTANCE;
                auto wos = t_interface.sample(wo, lcg(seed), make_float2(lcg(seed), lcg(seed)), mode);
                auto wis = t_interface.sample(wi, lcg(seed), make_float2(lcg(seed), lcg(seed)), reverse_mode);
                $if(!wos.eval.f.is_zero() & wos.eval.pdf > 0.f & !wis.eval.f.is_zero() & wis.eval.pdf > 0.f) {
                    auto rs = r_interface.sample(-wos.wi, lcg(seed), make_float2(lcg(seed), lcg(seed)), mode);
                    $if(!rs.eval.f.is_zero() & rs.eval.pdf > 0.f) {
                        auto r_pdf = r_interface.evaluate(-wos.wi, -wis.wi, mode).pdf;
                        auto wt = power_heuristic(1u, wis.eval.pdf, 1u, r_pdf);
                        pdf_sum += wt * r_pdf;
                        auto t_pdf = t_interface.evaluate(-rs.wi, wi, mode).pdf;
                        wt = power_heuristic(1u, rs.eval.pdf, 1u, t_pdf);
                        pdf_sum += wt * t_pdf;
                    };
                };
            }
            $else {
                auto ti_interface = TopOrBottom(function_bottom, ctx_wrapper_bottom, function_top, ctx_wrapper_top, swl, time, entered_top);
                auto to_interface = TopOrBottom(function_top, ctx_wrapper_top, function_bottom, ctx_wrapper_bottom, swl, time, entered_top);
                auto reverse_mode = mode == TransportMode::IMPORTANCE ? TransportMode::RADIANCE : TransportMode::IMPORTANCE;
                auto wos = to_interface.sample(wo, lcg(seed), make_float2(lcg(seed), lcg(seed)), mode);
                auto wis = ti_interface.sample(wi, lcg(seed), make_float2(lcg(seed), lcg(seed)), reverse_mode);
                $if(wos.eval.f.is_zero() | wos.eval.pdf <= 0.f |
                    wis.eval.f.is_zero() | wis.eval.pdf <= 0.f) { $continue; };
                pdf_sum += .5f * (to_interface.evaluate(wo, -wis.wi, mode).pdf +
                                  ti_interface.evaluate(-wos.wi, wi, mode).pdf);
            };
        };
        return {.f = f / Float(ctx->samples),
                .pdf = lerp(1.f / (4.f * pi), pdf_sum / Float(ctx->samples), 0.9f)};
    }
    [[nodiscard]] Surface::Sample sample(
        const Surface::FunctionContext *ctx_wrapper, const SampledWavelengths &swl, Expr<float> time,
        Expr<float3> wo, Expr<float> u_lobe, Expr<float2> u, TransportMode mode) const noexcept override {
        auto ctx = ctx_wrapper->data<LayeredSurfaceInstance::LayeredSurfaceContext>();
        auto &it = ctx->it;

        auto function_top = ctx_wrapper->nested("top").function(0u);
        auto ctx_wrapper_top = ctx_wrapper->nested("top").context(0u);
        auto function_bottom = ctx_wrapper->nested("bottom").function(0u);
        auto ctx_wrapper_bottom = ctx_wrapper->nested("bottom").context(0u);

        auto wo_local = it.shading().world_to_local(wo);
        auto entered_top = wo_local.z > 0.f;
        auto b_surf = TopOrBottom(function_top, ctx_wrapper_top, function_bottom, ctx_wrapper_bottom, swl, time, entered_top);
        auto bs = b_surf.sample(wo, u_lobe, u, mode);
        auto s = Surface::Sample::zero(swl.dimension());
        $if(!bs.eval.f.is_zero() & bs.eval.pdf != 0.f) {
            auto wi_local = it.shading().world_to_local(bs.wi);
            $if(same_hemisphere(wi_local, wo_local)) {
                s = bs;
            }
            $else {
                auto w = bs.wi;
                auto w_local = it.shading().world_to_local(bs.wi);
                auto seed = xxhash32(make_uint4(as<UInt3>(make_float3(u, u_lobe)), xxhash32(as<UInt3>(wo))));
                auto f = bs.eval.f;
                auto pdf = bs.eval.pdf;
                auto z = ite(entered_top, ctx->thickness, 0.f);
                HGPhaseFunction phase(ctx->g);
                $for(depth, ctx->max_depth) {
                    auto rr_beta = f.max() / pdf;
                    $if(depth > 3 & rr_beta < 0.25f) {
                        auto q = max(0.f, 1.f - rr_beta);
                        $if(lcg(seed) < q) { $break; };
                        pdf *= 1 - q;
                    };
                    $if(w_local.z == 0.f) { $break; };
                    $if(!ctx->albedo.is_zero()) {
                        auto sigma_t = 1.f;
                        auto dz = -log(1.f - lcg(seed)) / (sigma_t / abs(w_local.z));
                        auto zp = ite(w_local.z > 0.f, z + dz, z - dz);
                        $if(z == zp) { $break; };
                        $if(0.f < zp & zp < ctx->thickness) {
                            auto ps = phase.Sample_p(-w_local, make_float2(lcg(seed), lcg(seed)));
                            $if(ps.pdf <= 0.f) { $break; };
                            f *= ctx->albedo * ps.p;
                            pdf *= ps.pdf;
                            w = ps.wi;
                            w_local = it.shading().world_to_local(w);
                            z = zp;
                            $continue;
                        };
                        z = clamp(zp, 0.f, ctx->thickness);
                    }
                    $else {
                        z = ite(z == ctx->thickness, 0.f, ctx->thickness);
                        f *= Tr(ctx->thickness, w_local);
                    };
                    auto interface = TopOrBottom(function_bottom, ctx_wrapper_bottom, function_top, ctx_wrapper_top, swl, time, z == 0.f);
                    auto uc = lcg(seed);
                    auto u = make_float2(lcg(seed), lcg(seed));
                    auto bs = interface.sample(-w, uc, u, mode);
                    $if(bs.eval.f.is_zero() | bs.eval.pdf <= 0.f) { $break; };
                    f *= bs.eval.f;
                    pdf *= bs.eval.pdf;
                    w = bs.wi;
                    w_local = it.shading().world_to_local(w);
                    $if((bs.event & Surface::event_transmit) != 0u) {
                        s = Surface::Sample{
                            .eval = {.f = f, .pdf = pdf},
                            .wi = w,
                            .event = ite(same_hemisphere(w_local, wo_local),
                                         Surface::event_reflect,
                                         ite(w_local.z > 0.f, Surface::event_exit, Surface::event_enter))};
                        $break;
                    };
                };
            };
        };
        return s;
    }
};

uint LayeredSurfaceInstance::make_closure(
    PolymorphicClosure<Surface::Function> &closure,
    luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl,
    Expr<float3> wo, Expr<float> eta_i, Expr<float> time) const noexcept {
    auto thickness = _thickness ? max(_thickness->evaluate(*it, swl, time).x, std::numeric_limits<float>::min()) : 1e-2f;
    auto g = _g ? _g->evaluate(*it, swl, time).x : 0.f;
    auto [albedo, _] = _albedo ? _albedo->evaluate_albedo_spectrum(*it, swl, time) : Spectrum::Decode::one(swl.dimension());
    auto max_depth = node<LayeredSurface>()->max_depth();
    auto samples = node<LayeredSurface>()->samples();

    auto ctx = LayeredSurfaceContext{
        .it = *it,
        .thickness = thickness,
        .g = g,
        .albedo = albedo,
        .max_depth = max_depth,
        .samples = samples};

    auto [tag, slot] = closure.register_instance<LayeredSurfaceFunction>(node()->closure_identifier());
    slot->create_data(std::move(ctx));
    slot->create_nested("top");
    slot->create_nested("bottom");
    _top->make_closure(slot->nested("top"), it, swl, wo, eta_i, time);
    auto eta_top = slot->nested("top").function(0u)->eta(slot->nested("top").context(0u), swl, time);
    _bottom->make_closure(slot->nested("bottom"), it, swl, wo, eta_top.value_or(1.f), time);// FIXME: eta_i is wrong
    return tag;
}

//using TwoSidedNormalMapOpacityLayeredSurface = TwoSidedWrapper<NormalMapWrapper<OpacitySurfaceWrapper<
//    LayeredSurface, LayeredSurfaceInstance, LayeredSurfaceClosure>>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::LayeredSurface)
