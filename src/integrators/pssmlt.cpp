//
// Created by Mike Smith on 2022/11/9.
//

#include <util/u64.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

struct PrimarySample {
    float value;
    float value_backup;
    uint2 last_modification_iteration;
    uint2 modification_backup;
};

}// namespace luisa::render

// clang-format off
LUISA_STRUCT(luisa::render::PrimarySample,
             value, value_backup, last_modification_iteration, modification_backup) {
    void backup() noexcept {
        value_backup = value;
        modification_backup = last_modification_iteration;
    }
    void restore() noexcept {
        value = value_backup;
        last_modification_iteration = modification_backup;
    }
    void backup_if(luisa::compute::Expr<bool> p) noexcept {
        value_backup = ite(p, value, value_backup);
        modification_backup = ite(p, last_modification_iteration, modification_backup);
    }
    void restore_if(luisa::compute::Expr<bool> p) noexcept {
        value = ite(p, value_backup, value);
        last_modification_iteration = ite(p, modification_backup, last_modification_iteration);
    }
};
// clang-format on

namespace luisa::render {

using namespace compute;

class PCG32 {

private:
    static constexpr auto default_state = 0x853c49e6748fea9bull;
    static constexpr auto default_stream = 0xda3e39cb94b95bdbull;
    static constexpr auto mult = 0x5851f42d4c957f2dull;

private:
    U64 _state;
    U64 _inc;

public:
    // clang-format off
    PCG32() noexcept : _state{default_state}, _inc{default_stream} {}
    PCG32(U64 state, U64 inc) noexcept : _state{state}, _inc{inc} {}
    explicit PCG32(U64 seq_index) noexcept { set_sequence(seq_index); }
    explicit PCG32(Expr<uint> seq_index) noexcept { set_sequence(U64{seq_index}); }
    // clang-format on
    [[nodiscard]] auto uniform_uint() noexcept {
        auto oldstate = _state;
        _state = oldstate * U64{mult} + _inc;
        auto xorshifted = (((oldstate >> 18u) ^ oldstate) >> 27u).lo();
        auto rot = (oldstate >> 59u).lo();
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }
    void set_sequence(U64 init_seq) noexcept {
        _state = U64{0u};
        _inc = (init_seq << 1u) | 1u;
        static_cast<void>(uniform_uint());
        _state = _state + U64{default_state};
        static_cast<void>(uniform_uint());
    }
    [[nodiscard]] auto uniform_float() noexcept {
        return min(one_minus_epsilon, uniform_uint() * 0x1p-32f);
    }
    [[nodiscard]] auto state() const noexcept { return _state; }
    [[nodiscard]] auto inc() const noexcept { return _inc; }
};

class PSSMLTSampler {

public:
    struct State {
        PCG32 rng;
        U64 current_iteration;
        Bool large_step;
        U64 last_large_step_iteration;
        UInt sample_index;
        Local<PrimarySample> primary_samples;

        [[nodiscard]] static auto create(uint pss_dim, Expr<uint> rng_sequence) noexcept {
            return luisa::make_unique<State>(State{
                .rng = PCG32{rng_sequence},
                .current_iteration = U64{0u},
                .large_step = true,
                .last_large_step_iteration = U64{0u},
                .sample_index = UInt{0u},
                .primary_samples = Local<PrimarySample>{pss_dim}});
        }
    };

private:
    uint _mutations_per_pixel;
    float _sigma;
    float _large_step_probability;
    uint _pss_dimension;
    luisa::unique_ptr<State> _state;

private:
    [[nodiscard]] static auto _erf_inv(Expr<float> x) noexcept {
        static Callable impl = [](Float x) noexcept {
            auto w = def(0.f);
            auto p = def(0.f);
            x = clamp(x, -.99999f, .99999f);
            w = -log((1.f - x) * (1.f + x));
            $if(w < 5.f) {
                w = w - 2.5f;
                p = 2.81022636e-08f;
                p = 3.43273939e-07f + p * w;
                p = -3.5233877e-06f + p * w;
                p = -4.39150654e-06f + p * w;
                p = 0.00021858087f + p * w;
                p = -0.00125372503f + p * w;
                p = -0.00417768164f + p * w;
                p = 0.246640727f + p * w;
                p = 1.50140941f + p * w;
            }
            $else {
                w = sqrt(w) - 3.f;
                p = -0.000200214257f;
                p = 0.000100950558f + p * w;
                p = 0.00134934322f + p * w;
                p = -0.00367342844f + p * w;
                p = 0.00573950773f + p * w;
                p = -0.0076224613f + p * w;
                p = 0.00943887047f + p * w;
                p = 1.00167406f + p * w;
                p = 2.83297682f + p * w;
            };
            return p * x;
        };
        return impl(x);
    }

    void _ensure_ready(Expr<uint> index) noexcept {
        auto &Xi = _state->primary_samples[index];
        // Reset Xi if a large step took place in the meantime
        auto needs_reset = U64{Xi.last_modification_iteration} < _state->last_large_step_iteration;
        Xi.value = ite(needs_reset, _state->rng.uniform_float(), Xi.value);
        Xi.last_modification_iteration = ite(needs_reset, _state->last_large_step_iteration.bits(), Xi.last_modification_iteration);
        // Apply remaining sequence of mutations to _sample_
        Xi->backup();
        $if(_state->large_step) {
            Xi.value = _state->rng.uniform_float();
        }
        $else {
            auto nSmall = (_state->current_iteration - U64{Xi.last_modification_iteration}).lo();
            // Apply _nSmall_ small step mutations
            // Sample the standard normal distribution N(0, 1)
            auto normalSample = sqrt_two * _erf_inv(2.f * _state->rng.uniform_float() - 1.f);
            // Compute the effective standard deviation and apply perturbation to Xi
            auto effSigma = _sigma * sqrt(cast<float>(nSmall));
            Xi.value += normalSample * effSigma;
            Xi.value -= floor(Xi.value);
        };
        Xi.last_modification_iteration = _state->current_iteration.bits();
    }

public:
    PSSMLTSampler(uint mutations_per_pixel, float sigma,
                  float large_step_prob, uint pss_dim) noexcept
        : _mutations_per_pixel{mutations_per_pixel},
          _sigma{sigma},
          _large_step_probability{large_step_prob},
          _pss_dimension{pss_dim} {}

    void reset(Expr<uint> rng_sequence) noexcept {
        _state = State::create(_pss_dimension, rng_sequence);
    }

    void accept() noexcept {
        _state->last_large_step_iteration = ite(
            _state->large_step,
            _state->current_iteration,
            _state->last_large_step_iteration);
    }

    void reject() noexcept {
        for (auto i = 0u; i < _pss_dimension; i++) {
            auto &sample = _state->primary_samples[i];
            sample->restore_if(U64{sample.last_modification_iteration} == _state->current_iteration);
        }
        _state->current_iteration = _state->current_iteration - 1u;
    }

    void start_stream() noexcept { _state->sample_index = 0u; }

    [[nodiscard]] auto generate_1d() noexcept {
        _ensure_ready(_state->sample_index);
        return _state->primary_samples[_state->sample_index].value;
    }

    [[nodiscard]] auto generate_2d() noexcept {
        auto x = generate_1d();
        auto y = generate_1d();
        return make_float2(x, y);
    }

    void start_iteration() noexcept {
        _state->current_iteration = _state->current_iteration + 1u;
        _state->large_step = _state->rng.uniform_float() < _large_step_probability;
    }
};

class PSSMLT final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    // clang-format off
    PSSMLT(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)} {}
    // clang-format on
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PSSMLTInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        Instance::_render_one_camera(command_buffer, camera);
    }

    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time) const noexcept override {

        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension()};

        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        $for(depth, node<PSSMLT>()->max_depth()) {

            // trace
            auto it = pipeline().geometry()->intersect(ray);

            // miss
            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if(it->shape()->has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                };
            }

            $if(!it->shape()->has_surface()) { $break; };

            // sample one light
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            Light::Sample light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.ray);

            // evaluate material
            auto surface_tag = it->shape()->surface_tag();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto eta = def(1.f);
            auto eta_scale = def(1.f);
            auto alpha_skip = def(false);
            auto wo = -ray->direction();
            auto surface_sample = Surface::Sample::zero(swl.dimension());
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(*it, swl, 1.f, time);

                // apply roughness map
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(!alpha_skip) {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }
                    // direct lighting
                    $if(light_sample.eval.pdf > 0.0f & !occluded) {
                        auto wi = light_sample.ray->direction();
                        auto eval = closure->evaluate(wo, wi);
                        auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                 light_sample.eval.pdf;
                        Li += w * beta * eval.f * light_sample.eval.L;
                    };
                    // sample material
                    surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                    eta = closure->eta().value_or(1.f);
                };
            });

            $if(alpha_skip) {
                ray = it->spawn_ray(ray->direction());
                pdf_bsdf = 1e16f;
            }
            $else {
                ray = it->spawn_ray(surface_sample.wi);
                pdf_bsdf = surface_sample.eval.pdf;
                auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                beta *= w * surface_sample.eval.f;
                // apply eta scale
                $switch(surface_sample.event) {
                    $case(Surface::event_enter) { eta_scale = sqr(eta); };
                    $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                };
                // rr
                beta = zero_if_any_nan(beta);
                $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
                auto rr_depth = node<PSSMLT>()->rr_depth();
                auto rr_threshold = node<PSSMLT>()->rr_threshold();
                auto q = max(beta.max() * eta_scale, .05f);
                $if(depth + 1u >= rr_depth) {
                    auto u = sampler()->generate_1d();
                    $if(q < rr_threshold & u >= q) { $break; };
                    beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
                };
            };
        };
        return spectrum->srgb(swl, Li);
    }
};

luisa::unique_ptr<Integrator::Instance> PSSMLT::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<PSSMLTInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PSSMLT)
