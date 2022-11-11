//
// Created by Mike Smith on 2022/11/9.
//

#include <util/u64.h>
#include <util/rng.h>
#include <util/sampling.h>
#include <util/colorspace.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/display.h>

namespace luisa::render {

struct PrimarySample {
    float value;
    float value_backup;
    uint2 last_modification_iteration;
    uint2 modification_backup;

    static constexpr auto size_words = 6u;
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

class PSSMLTSampler {

public:
    struct State {
        UInt rng_state;
        U64 current_iteration;
        Bool large_step;
        U64 last_large_step_iteration;
        Local<PrimarySample> primary_samples;
        UInt sample_index;

        [[nodiscard]] static auto create(uint pss_dim, Expr<uint> rng_sequence) noexcept {
            return luisa::make_unique<State>(State{
                .rng_state = xxhash32(rng_sequence),
                .current_iteration = U64{0u},
                .large_step = true,
                .last_large_step_iteration = U64{0u},
                .primary_samples = Local<PrimarySample>{pss_dim},
                .sample_index = 0u});
        }

        [[nodiscard]] static auto size_words(uint pss_dim) noexcept {
            return 1u +// rng
                   2u +// current_iteration
                   1u +// large_step
                   2u +// last_large_step_iteration
                   pss_dim * PrimarySample::size_words;
        }

        [[nodiscard]] static auto load(BufferView<uint> state_buffer, Expr<uint> index, uint pss_dim) noexcept {
            auto offset = index * size_words(pss_dim);
            auto rng_state = state_buffer.read(offset + 0u);
            auto current_iteration = make_uint2(state_buffer.read(offset + 1u),
                                                state_buffer.read(offset + 2u));
            auto large_step = state_buffer.read(offset + 3u) != 0u;
            auto last_large_step_iteration = make_uint2(state_buffer.read(offset + 4u),
                                                        state_buffer.read(offset + 5u));
            auto primary_samples = Local<PrimarySample>{pss_dim};
            for (auto i = 0u; i < pss_dim; i++) {
                auto sample_offset = offset + 6u + i * PrimarySample::size_words;
                primary_samples[i].value = as<float>(state_buffer.read(sample_offset + 0u));
                primary_samples[i].value_backup = as<float>(state_buffer.read(sample_offset + 1u));
                primary_samples[i].last_modification_iteration = make_uint2(state_buffer.read(sample_offset + 2u),
                                                                            state_buffer.read(sample_offset + 3u));
                primary_samples[i].modification_backup = make_uint2(state_buffer.read(sample_offset + 4u),
                                                                    state_buffer.read(sample_offset + 5u));
            }
            return luisa::make_unique<State>(State{
                .rng_state = rng_state,
                .current_iteration = U64{current_iteration},
                .large_step = large_step,
                .last_large_step_iteration = U64{last_large_step_iteration},
                .primary_samples = primary_samples,
                .sample_index = 0u});
        }

        void save(BufferView<uint> state_buffer, Expr<uint> index) noexcept {
            auto offset = index * size_words(primary_samples.size());
            state_buffer.write(offset + 0u, rng_state);
            state_buffer.write(offset + 1u, current_iteration.bits().x);
            state_buffer.write(offset + 2u, current_iteration.bits().y);
            state_buffer.write(offset + 3u, ite(large_step, 1u, 0u));
            state_buffer.write(offset + 4u, last_large_step_iteration.bits().x);
            state_buffer.write(offset + 5u, last_large_step_iteration.bits().y);
            for (auto i = 0u; i < primary_samples.size(); i++) {
                auto sample_offset = offset + 6u + i * PrimarySample::size_words;
                state_buffer.write(sample_offset + 0u, as<uint>(primary_samples[i].value));
                state_buffer.write(sample_offset + 1u, as<uint>(primary_samples[i].value_backup));
                state_buffer.write(sample_offset + 2u, primary_samples[i].last_modification_iteration.x);
                state_buffer.write(sample_offset + 3u, primary_samples[i].last_modification_iteration.y);
                state_buffer.write(sample_offset + 4u, primary_samples[i].modification_backup.x);
                state_buffer.write(sample_offset + 5u, primary_samples[i].modification_backup.y);
            }
        }
    };

private:
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
        Xi.value = ite(needs_reset, lcg(_state->rng_state), Xi.value);
        Xi.last_modification_iteration = ite(needs_reset, _state->last_large_step_iteration.bits(), Xi.last_modification_iteration);
        // Apply remaining sequence of mutations to _sample_
        Xi->backup();
        $if(_state->large_step) {
            Xi.value = lcg(_state->rng_state);
        }
        $else {
            auto nSmall = (_state->current_iteration - U64{Xi.last_modification_iteration}).lo();
            // Apply _nSmall_ small step mutations
            // Sample the standard normal distribution N(0, 1)
            auto normalSample = sqrt_two * _erf_inv(2.f * lcg(_state->rng_state) - 1.f);
            // Compute the effective standard deviation and apply perturbation to Xi
            auto effSigma = _sigma * sqrt(cast<float>(nSmall));
            Xi.value += normalSample * effSigma;
            Xi.value -= floor(Xi.value);
        };
        Xi.last_modification_iteration = _state->current_iteration.bits();
    }

public:
    PSSMLTSampler(float sigma, float large_step_prob, uint pss_dim) noexcept
        : _sigma{sigma},
          _large_step_probability{large_step_prob},
          _pss_dimension{pss_dim} {}

    void create(Expr<uint> rng_sequence) noexcept {
        _state = State::create(_pss_dimension, rng_sequence);
    }

    void load(BufferView<uint> state_buffer, Expr<uint> index) noexcept {
        _state = State::load(state_buffer, index, _pss_dimension);
    }

    void save(BufferView<uint> state_buffer, Expr<uint> index) noexcept {
        _state->save(state_buffer, index);
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

    [[nodiscard]] auto generate_1d() noexcept {
        auto index = _state->sample_index;
        _state->sample_index += 1u;
        _ensure_ready(index);
        return _state->primary_samples[index].value;
    }

    [[nodiscard]] auto generate_2d() noexcept {
        auto x = generate_1d();
        auto y = generate_1d();
        return make_float2(x, y);
    }

    void start_iteration() noexcept {
        _state->current_iteration = _state->current_iteration + 1u;
        _state->large_step = lcg(_state->rng_state) < _large_step_probability;
    }
};

class PSSMLT final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _bootstrap_samples;
    uint _chains;
    float _large_step_probability;
    float _sigma;
    luisa::unique_ptr<PSSMLTSampler> _sampler;

public:
    PSSMLT(Scene *scene, const SceneNodeDesc *desc)
    noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _bootstrap_samples{std::max(desc->property_uint_or_default("bootstrap_samples", 1024u * 1024u), 1u)},
          _chains{std::max(desc->property_uint_or_default("chains", 256u * 1024u), 1u)},
          _large_step_probability{std::clamp(
              desc->property_float_or_default(
                  "large_step_probability", lazy_construct([desc] {
                      return desc->property_float_or_default("large_step", .3f);
                  })),
              0.f, 1.f)},
          _sigma{std::max(desc->property_float_or_default("sigma", 1e-2f), 1e-4f)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto bootstrap_samples() const noexcept { return _bootstrap_samples; }
    [[nodiscard]] auto chains() const noexcept { return _chains; }
    [[nodiscard]] auto large_step_probability() const noexcept { return _large_step_probability; }
    [[nodiscard]] auto sigma() const noexcept { return _sigma; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PSSMLTInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

private:
    [[nodiscard]] auto _compute_pss_dimension(const Camera *camera) const noexcept {
        auto max_depth = node<PSSMLT>()->max_depth();
        auto rr_depth = std::clamp(node<PSSMLT>()->rr_depth(), 1u, max_depth - 1u);
        auto dim = 2u;// pixel
        if (camera->requires_lens_sampling()) { dim += 2u; }
        if (!pipeline().spectrum()->node()->is_fixed()) { dim += 1u; }
        for (auto depth = 0u; depth < max_depth; depth++) {
            dim += 1u +// light selection
                   2u +// light area
                   1u +// BSDF lobe
                   2u; // BSDF direction
            if (depth + 1u >= rr_depth) { dim += 1u /* RR */; }
        }
        return dim;
    }

    [[nodiscard]] static auto _s(Expr<float3> L) noexcept {
        //                return max(srgb_to_cie_y(L), 0.f);
        auto v = clamp(L, 0.f, 1e3f);
        return v.x + v.y + v.z;
        //        return max(max(L.x, L.y), max(L.z, 0.f));
    }

    [[nodiscard]] auto Li(PSSMLTSampler &sampler,
                          const Camera::Instance *camera,
                          Expr<float> time,
                          UInt &rng_state) const noexcept {

        auto res = make_float2(camera->film()->node()->resolution());
        auto p = sampler.generate_2d() * res;
        auto pixel_id = make_uint2(clamp(p, 0.f, res - 1.f));
        auto u_filter = fract(p);
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler.generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : lcg(rng_state));
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
            auto u_light_selection = sampler.generate_1d();
            auto u_light_surface = sampler.generate_2d();
            Light::Sample light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.ray);

            // evaluate material
            auto surface_tag = it->shape()->surface_tag();
            auto u_lobe = sampler.generate_1d();
            auto u_bsdf = sampler.generate_2d();
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
                    auto u = sampler.generate_1d();
                    $if(q < rr_threshold & u >= q) { $break; };
                    beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
                };
            };
        };
        return std::make_pair(pixel_id, spectrum->srgb(swl, Li));
    }

private:
    [[nodiscard]] auto _bootstrap(CommandBuffer &command_buffer,
                                  Camera::Instance *camera, float initial_time) noexcept {
        auto bootstrap_count = node<PSSMLT>()->bootstrap_samples();
        auto bootstrap_weights = pipeline().device().create_buffer<float>(bootstrap_count);
        command_buffer << synchronize();

        Clock clk;
        LUISA_INFO("PSSMLT: compiling bootstrap kernel.");
        auto bootstrap = pipeline().device().compile<1u>([&](Float time) noexcept {
            auto bootstrap_id = dispatch_x();
            PSSMLTSampler sampler{node<PSSMLT>()->sigma(),
                                  node<PSSMLT>()->large_step_probability(),
                                  _compute_pss_dimension(camera->node())};
            sampler.create(bootstrap_id);
            auto rng_state = xxhash32(make_uint2(bootstrap_id, 0xdeadbeefu));
            auto [_, L] = Li(sampler, camera, time, rng_state);
            bootstrap_weights.write(bootstrap_id, _s(L));
        });
        LUISA_INFO("PSSMLT: running bootstrap kernel.");
        luisa::vector<float> bw(bootstrap_count);
        command_buffer << bootstrap(initial_time).dispatch(bootstrap_count)
                       << bootstrap_weights.copy_to(bw.data())
                       << synchronize();
        LUISA_INFO("PSSMLT: Generated {} bootstrap sample(s) in {} ms.",
                   bootstrap_count, clk.toc());
        auto b = 0.;
        for (auto w : bw) { b += w; }
        b /= bootstrap_count;
        LUISA_INFO("PSSMLT: normalization factor is {}.", b);
        auto [alias_table, _] = create_alias_table(bw);
        auto bootstrap_sampling_table = pipeline().device().create_buffer<AliasEntry>(alias_table.size());
        command_buffer << bootstrap_sampling_table.copy_from(alias_table.data())
                       << commit();
        return std::make_pair(std::move(bootstrap_sampling_table), b);
    }

    void _render(CommandBuffer &command_buffer, Camera::Instance *camera,
                 luisa::span<const Camera::ShutterSample> shutter_samples,
                 Buffer<AliasEntry> bootstrap_sampling_table, double b) noexcept {
        auto pss_dim = _compute_pss_dimension(camera->node());
        auto sigma = node<PSSMLT>()->sigma();
        auto large_step_prob = node<PSSMLT>()->large_step_probability();
        auto chains = node<PSSMLT>()->chains();
        LUISA_INFO("PSSMLT: rendering {} chain(s) with {} "
                   "primary sample(s) of {} dimension(s) per pixel.",
                   chains, camera->node()->spp(), pss_dim);

        auto resolution = camera->film()->node()->resolution();
        auto pixel_count = resolution.x * resolution.y;

        auto state_buffer = pipeline().device().create_buffer<uint>(
            chains * PSSMLTSampler::State::size_words(pss_dim));
        auto radiance_buffer = pipeline().device().create_buffer<float3>(chains);
        auto position_buffer = pipeline().device().create_buffer<uint2>(chains);
        auto shutter_weight_buffer = pipeline().device().create_buffer<float>(chains);
        auto accumulate_buffer = pipeline().device().create_buffer<float>(pixel_count * 3u);
        auto statistics_buffer = pipeline().device().create_buffer<uint>(4u);

        Clock clk;
        LUISA_INFO("PSSMLT: compiling create_chains kernel...");
        auto create_chains = pipeline().device().compile<1u>([&](Float time, Float shutter_weight) noexcept {
            auto chain_id = dispatch_x();
            auto seed = xxhash32(make_uint2(chain_id, 0xabadfaceu));
            auto u = lcg(seed);
            auto [bootstrap_id, _] = sample_alias_table(bootstrap_sampling_table,
                                                        static_cast<uint>(bootstrap_sampling_table.size()), u);
            PSSMLTSampler sampler{sigma, large_step_prob, pss_dim};
            sampler.create(bootstrap_id);
            auto [p, L] = Li(sampler, camera, time, seed);
            position_buffer.write(chain_id, p);
            radiance_buffer.write(chain_id, L);
            shutter_weight_buffer.write(chain_id, shutter_weight);
            sampler.save(state_buffer, chain_id);
            $if(dispatch_x() < 4u) { statistics_buffer.write(dispatch_x(), 0u); };
        });
        LUISA_INFO("PSSMLT: compiled create_chains kernel in {} ms.", clk.toc());

        clk.tic();
        LUISA_INFO("PSSMLT: compiling render kernel...");
        auto render = pipeline().device().compile<1u>([&](UInt mutation_index, Float time, Float shutter_weight) noexcept {
            set_block_size(8u, 8u, 1u);
            auto chain_id = dispatch_id().x;
            auto seed = xxhash32(make_uint3(chain_id, mutation_index, 0xfacebeefu));
            PSSMLTSampler sampler{sigma, large_step_prob, pss_dim};
            sampler.load(state_buffer, chain_id);
            auto curr_p = position_buffer.read(chain_id);
            auto curr_L = radiance_buffer.read(chain_id);
            auto curr_w = shutter_weight_buffer.read(chain_id);
            sampler.start_iteration();
            auto proposed = Li(sampler, camera, time, seed);
            auto proposed_p = proposed.first;
            auto proposed_L = proposed.second;
            auto curr_y = _s(curr_L);
            auto proposed_y = _s(proposed_L);
            // Compute acceptance probability for proposed sample
            auto accept = min(1.f, proposed_y / curr_y);
            // Splat both current and proposed samples to _film_
            auto accum = [&accumulate_buffer, resolution](Expr<uint2> p, Expr<float3> L) noexcept {
                auto offset = (p.y * resolution.x + p.x) * 3u;
                $if(!any(isnan(L))) {
                    for (auto i = 0u; i < 3u; i++) {
                        accumulate_buffer.atomic(offset + i).fetch_add(L[i]);
                    }
                };
            };
            auto record = [&statistics_buffer](Expr<uint> index) noexcept {
                auto old = statistics_buffer.atomic(index * 2u + 0u).fetch_add(1u);
                $if(old == ~0u) { statistics_buffer.atomic(index * 2u + 1u).fetch_add(1u); };
            };
            $if(accept > 0.f) {
                accum(proposed_p, (accept * shutter_weight / proposed_y) * proposed_L);
            };
            accum(curr_p, ((1.f - accept) * curr_w / curr_y) * curr_L);
            // Accept or reject the proposal
            auto u = lcg(seed);
            $if(u < accept) {
                position_buffer.write(chain_id, proposed_p);
                radiance_buffer.write(chain_id, proposed_L);
                shutter_weight_buffer.write(chain_id, shutter_weight);
                sampler.accept();
                record(0u);
            }
            $else {
                sampler.reject();
                record(1u);
            };
            sampler.save(state_buffer, chain_id);
        });
        LUISA_INFO("PSSMLT: compiled render kernel in {} ms.", clk.toc());

        clk.tic();
        LUISA_INFO("PSSMLT: compiling clear kernel...");
        auto clear = pipeline().device().compile<1u>([&]() noexcept {
            auto pixel_id = dispatch_id().x;
            accumulate_buffer.write(pixel_id * 3u + 0u, 0.f);
            accumulate_buffer.write(pixel_id * 3u + 1u, 0.f);
            accumulate_buffer.write(pixel_id * 3u + 2u, 0.f);
        });
        LUISA_INFO("PSSMLT: compiled clear kernel in {} ms.", clk.toc());

        clk.tic();
        LUISA_INFO("PSSMLT: compiling accumulate kernel...");
        auto blit = pipeline().device().compile<2>([&](Float scale) {
            auto p = dispatch_id().xy();
            auto offset = (p.y * resolution.x + p.x) * 3u;
            auto L = make_float3(accumulate_buffer.read(offset + 0u),
                                 accumulate_buffer.read(offset + 1u),
                                 accumulate_buffer.read(offset + 2u));
            camera->film()->accumulate(p, scale * L);
        });
        LUISA_INFO("PSSMLT: compiled blit kernel in {} ms.", clk.toc());

        clk.tic();
        command_buffer << create_chains(shutter_samples.front().point.time,
                                        shutter_samples.front().point.weight)
                              .dispatch(chains)
                       << clear().dispatch(pixel_count)
                       << synchronize();
        LUISA_INFO("PSSMLT: created {} chain(s) in {} ms.", chains, clk.toc());

        clk.tic();
        LUISA_INFO("Rendering started.");
        ProgressBar progress;
        progress.update(0.);
        auto dispatch_count = 0ull;
        auto mutation_count = 0ull;
        auto mutation_per_chain_count = 0ull;
        auto total_mutations = static_cast<uint64_t>(camera->node()->spp()) * pixel_count;
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            auto mutations = static_cast<uint64_t>(s.spp) * pixel_count;
            auto mutations_per_chain = (mutations + chains - 1u) / chains;
            for (auto i = static_cast<uint64_t>(0u); i < mutations_per_chain; i++) {
                auto chains_to_dispatch = std::min((i + 1u) * chains, mutations) - i * chains;
                command_buffer << render(static_cast<uint>(mutation_per_chain_count++),
                                         s.point.time, s.point.weight)
                                      .dispatch(chains_to_dispatch);
                mutation_count += chains_to_dispatch;
                auto dispatches_per_commit =
                    display() && !display()->should_close() ?
                        node<ProgressiveIntegrator>()->display_interval() * std::max(pixel_count / chains, 1u) :
                        64u;
                if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                    auto p = static_cast<double>(mutation_count) / static_cast<double>(total_mutations);
                    auto effective_spp = p * camera->node()->spp();
                    auto statistics = luisa::make_shared<uint4>();
                    camera->film()->clear(command_buffer);
                    command_buffer << blit(static_cast<float>(b / effective_spp)).dispatch(resolution)
                                   << statistics_buffer.copy_to(statistics.get())
                                   << [statistics] {
                                          auto accepted = (static_cast<uint64_t>(statistics->y) << 32u) | statistics->x;
                                          auto rejected = (static_cast<uint64_t>(statistics->w) << 32u) | statistics->z;
                                          auto total = accepted + rejected;
                                          LUISA_INFO("PSSMLT: {}/{} mutation(s) accepted ({:.2f}%).",
                                                     accepted, total, static_cast<double>(accepted) / static_cast<double>(total) * 100.);
                                      };
                    dispatch_count = 0u;
                    if (display() && display()->update(command_buffer, static_cast<uint>(effective_spp))) {
                        progress.update(p);
                    } else {
                        command_buffer << [&progress, p] { progress.update(p); };
                    }
                }
            }
        }
        // final
        camera->film()->clear(command_buffer);
        command_buffer << blit(static_cast<float>(b / camera->node()->spp())).dispatch(resolution);
        command_buffer << synchronize();
        progress.done();

        auto render_time = clk.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        auto shutter_samples = camera->node()->shutter_samples();

        // bootstrap
        auto initial_time = shutter_samples.front().point.time;
        pipeline().update(command_buffer, initial_time);
        auto [bs, b] = _bootstrap(command_buffer, camera, initial_time);

        // perform actual rendering
        _render(command_buffer, camera, shutter_samples, std::move(bs), b);
    }
};

luisa::unique_ptr<Integrator::Instance> PSSMLT::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<PSSMLTInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PSSMLT)
