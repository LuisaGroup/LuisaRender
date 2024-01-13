//
// Created by Mike Smith on 2022/11/9.
//

#include <util/u64.h>
#include <util/rng.h>
#include <util/sampling.h>
#include <util/progress_bar.h>
#include <util/counter_buffer.h>
#include <base/pipeline.h>
#include <base/integrator.h>

namespace luisa::render {

struct alignas(8) PrimarySample {
    float value;
    float value_backup;
    uint2 last_modification;
    uint2 modification_backup;
};

}// namespace luisa::render

// clang-format off
LUISA_STRUCT(luisa::render::PrimarySample,
             value, value_backup,
             last_modification,
             modification_backup){
    void backup() noexcept {
        value_backup = value;
        modification_backup = last_modification;
    }
    void restore() noexcept {
        value = value_backup;
        last_modification = modification_backup;
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
        UInt chain_index;
        UInt sample_index;
        UInt initialized_dimensions;
    };

private:
    Device &_device;
    float _sigma;
    float _large_step_probability;
    luisa::unique_ptr<State> _state;

private:
    uint _chains{};
    uint _pss_dim{};
    Buffer<uint> _rng_buffer;
    Buffer<uint2> _current_iteration_buffer;
    Buffer<uint> _large_step_and_initialized_dimensions_buffer;
    Buffer<uint2> _last_large_step_iteration_buffer;
    Buffer<PrimarySample> _primary_sample_buffer;

public:
    void reset(CommandBuffer &command_buffer, uint chains, uint pss_dim) noexcept {
        LUISA_INFO("PSSMLT: Resetting sampler with {} chains and {} dimensions.", chains, pss_dim);
        _chains = chains;
        _pss_dim = pss_dim;
        LUISA_ASSERT(static_cast<uint64_t>(chains) * pss_dim <= ~0u,
                     "Too many primary samples.");
        command_buffer << synchronize();
        if (auto n = next_pow2(chains); n > _rng_buffer.size()) {
            _rng_buffer = _device.create_buffer<uint>(n);
            _current_iteration_buffer = _device.create_buffer<uint2>(n);
            _large_step_and_initialized_dimensions_buffer = _device.create_buffer<uint>(n);
            _last_large_step_iteration_buffer = _device.create_buffer<uint2>(n);
        }
        if (auto n = next_pow2(chains * pss_dim); n > _primary_sample_buffer.size()) {
            _primary_sample_buffer = _device.create_buffer<PrimarySample>(n);
        }
    }

private:
    [[nodiscard]] auto _primary_sample_index(Expr<uint> dim) const noexcept {
        // use the SoA layout to improve memory access locality
        return dim * _chains + _state->chain_index;
    }

    [[nodiscard]] auto _read_primary_sample(Expr<uint> dim) const noexcept {
        auto i = _primary_sample_index(dim);
        return _primary_sample_buffer->read(i);
    }

    void _write_primary_sample(Expr<uint> dim, Expr<PrimarySample> sample) noexcept {
        auto i = _primary_sample_index(dim);
        _primary_sample_buffer->write(i, sample);
    }

    [[nodiscard]] static auto _erf_inv(Expr<float> x) noexcept {
        static Callable impl = [](Float x) noexcept {
            auto w = def(0.f);
            auto p = def(0.f);
            x = clamp(x, -.99999f, .99999f);
            w = -log((1.f - x) * (1.f + x));
            $if(w < 5.f) {
                w = w - 2.5f;
                p = 2.81022636e-08f;
                p = fma(p, w, 3.43273939e-07f);
                p = fma(p, w, -3.5233877e-06f);
                p = fma(p, w, -4.39150654e-06f);
                p = fma(p, w, 0.00021858087f);
                p = fma(p, w, -0.00125372503f);
                p = fma(p, w, -0.00417768164f);
                p = fma(p, w, 0.246640727f);
                p = fma(p, w, 1.50140941f);
            }
            $else {
                w = sqrt(w) - 3.f;
                p = -0.000200214257f;
                p = fma(p, w, 0.000100950558f);
                p = fma(p, w, 0.00134934322f);
                p = fma(p, w, -0.00367342844f);
                p = fma(p, w, 0.00573950773f);
                p = fma(p, w, -0.0076224613f);
                p = fma(p, w, 0.00943887047f);
                p = fma(p, w, 1.00167406f);
                p = fma(p, w, 2.83297682f);
            };
            return p * x;
        };
        return impl(x);
    }

    [[nodiscard]] auto _sample(Expr<uint> index) noexcept {
        auto Xi = def<PrimarySample>();
        $if(_state->initialized_dimensions <= index) {// Initialize the sample
            Xi.value = 0.f;
            Xi.value_backup = 0.f;
            Xi.last_modification = make_uint2();
            Xi.modification_backup = make_uint2();
            _state->initialized_dimensions += 1u;
        }
        $else {// Load the sample
            Xi = _read_primary_sample(index);
        };
        // Reset Xi if a large step took place in the meantime
        $if(U64{Xi.last_modification} < _state->last_large_step_iteration) {
            Xi.value = lcg(_state->rng_state);
            Xi.last_modification = _state->last_large_step_iteration.bits();
        };
        // Apply remaining sequence of mutations to _sample_
        Xi->backup();
        $if(_state->large_step) {
            Xi.value = lcg(_state->rng_state);
        }
        $else {
            auto nSmall = (_state->current_iteration - U64{Xi.last_modification}).lo();
            // Apply _nSmall_ small step mutations
            // Sample the standard normal distribution N(0, 1)
            auto normalSample = sqrt_two * _erf_inv(2.f * lcg(_state->rng_state) - 1.f);
            // Compute the effective standard deviation and apply perturbation to Xi
            auto effSigma = _sigma * sqrt(cast<float>(nSmall));
            Xi.value = fract(Xi.value + normalSample * effSigma);
        };
        Xi.last_modification = _state->current_iteration.bits();
        // Store the sample
        _write_primary_sample(index, Xi);
        return Xi.value;
    }

public:
    PSSMLTSampler(Device &device, float sigma, float large_step_prob) noexcept
        : _device{device}, _sigma{sigma}, _large_step_probability{large_step_prob} {}

    void create(Expr<uint> chain_index, Expr<uint> rng_sequence) noexcept {
        _state = luisa::make_unique<State>(State{
            .rng_state = xxhash32(rng_sequence),
            .current_iteration = U64{0u},
            .large_step = true,
            .last_large_step_iteration = U64{0u},
            .chain_index = chain_index,
            .sample_index = 0u,
            .initialized_dimensions = 0u});
    }

    void load(Expr<uint> chain_index) noexcept {
        auto rng_state = _rng_buffer->read(chain_index);
        auto current_iteration = _current_iteration_buffer->read(chain_index);
        auto large_step_and_dimensions = _large_step_and_initialized_dimensions_buffer->read(chain_index);
        auto last_large_step_iteration = _last_large_step_iteration_buffer->read(chain_index);
        _state = luisa::make_unique<State>(State{
            .rng_state = rng_state,
            .current_iteration = U64{current_iteration},
            .large_step = (large_step_and_dimensions & 1u) != 0u,
            .last_large_step_iteration = U64{last_large_step_iteration},
            .chain_index = chain_index,
            .sample_index = 0u,
            .initialized_dimensions = large_step_and_dimensions >> 1u});
    }

    void save() noexcept {
        _rng_buffer->write(_state->chain_index, _state->rng_state);
        _current_iteration_buffer->write(_state->chain_index, _state->current_iteration.bits());
        _large_step_and_initialized_dimensions_buffer->write(
            _state->chain_index, ite(_state->large_step, 1u, 0u) | (_state->initialized_dimensions << 1u));
        _last_large_step_iteration_buffer->write(_state->chain_index, _state->last_large_step_iteration.bits());
    }

    void accept() noexcept {
        _state->last_large_step_iteration = ite(
            _state->large_step,
            _state->current_iteration,
            _state->last_large_step_iteration);
    }

    void reject() noexcept {
        $for(i, _state->initialized_dimensions) {
            auto sample = _read_primary_sample(i);
            $if(U64{sample.last_modification} == _state->current_iteration) {
                sample->restore();
                _write_primary_sample(i, sample);
            };
        };
        _state->current_iteration = _state->current_iteration - 1u;
    }

    [[nodiscard]] auto large_step() const noexcept {
        return _state->large_step;
    }

    [[nodiscard]] auto generate_1d() noexcept {
        auto x = _sample(_state->sample_index);
        _state->sample_index += 1u;
        return x;
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
    bool _statistics;

public:
    PSSMLT(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), .05f)},
          _bootstrap_samples{std::max(desc->property_uint_or_default("bootstrap_samples", 1024u * 1024u), 1u)},
          _chains{std::max(desc->property_uint_or_default("chains", 256u * 1024u), 1u)},
          _large_step_probability{std::clamp(
              desc->property_float_or_default(
                  "large_step_probability", lazy_construct([desc] {
                      return desc->property_float_or_default(
                          "large_step", lazy_construct([desc] {
                              return desc->property_float_or_default("p_large", .3f);
                          }));
                  })),
              0.f, 1.f)},
          _sigma{std::max(desc->property_float_or_default("sigma", 5e-3f), 1e-4f)},
          _statistics{desc->property_bool_or_default(
              "statistics", lazy_construct([desc] {
                  return desc->property_bool_or_default("stat", false);
              }))} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto bootstrap_samples() const noexcept { return _bootstrap_samples; }
    [[nodiscard]] auto chains() const noexcept { return _chains; }
    [[nodiscard]] auto large_step_probability() const noexcept { return _large_step_probability; }
    [[nodiscard]] auto sigma() const noexcept { return _sigma; }
    [[nodiscard]] auto enable_statistics() const noexcept { return _statistics; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PSSMLTInstance final : public ProgressiveIntegrator::Instance {

private:
    luisa::unique_ptr<PSSMLTSampler> _sampler;

public:
    PSSMLTInstance(Pipeline &ppl, CommandBuffer &cb, const PSSMLT *node) noexcept
        : ProgressiveIntegrator::Instance{ppl, cb, node},
          _sampler{luisa::make_unique<PSSMLTSampler>(
              ppl.device(), node->sigma(), node->large_step_probability())} {}

private:
    [[nodiscard]] auto _compute_pss_dimension(const Camera *camera) const noexcept {
        auto max_depth = node<PSSMLT>()->max_depth();
        auto rr_depth = node<PSSMLT>()->rr_depth();
        auto dim = 4u;// pixel and filter
        if (camera->requires_lens_sampling()) { dim += 2u; }
        for (auto depth = 0u; depth < max_depth; depth++) {
            dim += 1u +// light selection
                   2u +// light area
                   1u +// BSDF lobe
                   2u; // BSDF direction
            if (depth + 1u >= rr_depth) { dim += 1u /* RR */; }
        }
        return dim;
    }

    [[nodiscard]] static auto _s(Expr<float3> L, Expr<bool> is_light) noexcept {
        auto v = clamp(L, 0.f, ite(is_light, 1.f, 1e4f));
        return v.x + v.y + v.z;
    }

    [[nodiscard]] auto Li(PSSMLTSampler &sampler,
                          UInt &rng_state,
                          const Camera::Instance *camera,
                          Expr<float> time) const noexcept {

        auto res = make_float2(camera->film()->node()->resolution());
        auto p = sampler.generate_2d() * res;
        auto pixel_id = make_uint2(clamp(p, 0.f, res - 1.f));
        auto u_filter = sampler.generate_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler.generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : lcg(rng_state));
        SampledSpectrum beta{swl.dimension(), camera_weight};
        SampledSpectrum Li{swl.dimension()};
        auto is_visible_light = def(false);

        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        $for(depth, node<PSSMLT>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            // miss
            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    is_visible_light |= depth == 0u;
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    is_visible_light |= depth == 0u;
                };
            }

            $if(!it->shape().has_surface()) { $break; };

            // sample one light
            auto u_light_selection = sampler.generate_1d();
            auto u_light_surface = sampler.generate_2d();
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto u_lobe = sampler.generate_1d();
            auto u_bsdf = sampler.generate_2d();
            auto eta_scale = def(1.f);
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                if (auto dispersive = closure->is_dispersive()) {
                    $if(*dispersive) { swl.terminate_secondary(); };
                }
                // direct lighting
                $if(light_sample.eval.pdf > 0.0f & !occluded) {
                    auto wi = light_sample.shadow_ray->direction();
                    auto eval = closure->evaluate(wo, wi);
                    auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                             light_sample.eval.pdf;
                    Li += w * beta * eval.f * light_sample.eval.L;
                };
                // sample material
                auto surface_sample = closure->sample(wo, u_lobe, u_bsdf);
                ray = it->spawn_ray(surface_sample.wi);
                pdf_bsdf = surface_sample.eval.pdf;
                auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                beta *= w * surface_sample.eval.f;
                // apply eta scale
                auto eta = closure->eta().value_or(1.f);
                $switch(surface_sample.event) {
                    $case(Surface::event_enter) { eta_scale = sqr(eta); };
                    $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                };
            });
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
        return std::make_tuple(pixel_id, spectrum->srgb(swl, Li), is_visible_light);
    }

private:
    [[nodiscard]] auto _bootstrap(CommandBuffer &command_buffer,
                                  Camera::Instance *camera, float initial_time) noexcept {
        auto bootstrap_count = node<PSSMLT>()->bootstrap_samples();
        auto bootstrap_weights = pipeline().device().create_buffer<float>(bootstrap_count);
        command_buffer << synchronize();

        Clock clk;
        LUISA_INFO("PSSMLT: compiling bootstrap kernel.");
        auto bootstrap = pipeline().device().compile<1u>([&](UInt bootsrape_offset, Float time) noexcept {
            auto chain_id = dispatch_x();
            auto bootstrap_id = chain_id + bootsrape_offset;
            auto seed = xxhash32(make_uint2(bootstrap_id, 0xdeadbeefu));
            _sampler->create(chain_id, bootstrap_id);
            auto [_, L, is_light] = Li(*_sampler, seed, camera, time);
            bootstrap_weights->write(bootstrap_id, _s(L, is_light));
        });
        LUISA_INFO("PSSMLT: running bootstrap kernel.");
        luisa::vector<float> bw(bootstrap_count);
        auto chains = node<PSSMLT>()->chains();
        auto dispatches = (bootstrap_count + chains - 1u) / chains;
        for (auto i = 0u; i < dispatches; i++) {
            auto chains_to_dispatch = std::min((i + 1u) * chains, bootstrap_count) - i * chains;
            command_buffer << bootstrap(i * chains, initial_time).dispatch(chains_to_dispatch);
        }
        command_buffer << bootstrap_weights.copy_to(bw.data()) << synchronize();
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
        auto p_large = node<PSSMLT>()->large_step_probability();
        auto chains = node<PSSMLT>()->chains();
        LUISA_INFO("PSSMLT: rendering {} chain(s) with {} "
                   "sample(s) of {} PSS dimension(s) per pixel.",
                   chains, camera->node()->spp(), pss_dim);

        auto resolution = camera->film()->node()->resolution();
        auto pixel_count = resolution.x * resolution.y;
        auto radiance_and_contribution_buffer = pipeline().device().create_buffer<float4>(chains);
        auto position_buffer = pipeline().device().create_buffer<uint2>(chains);
        auto accumulate_buffer = pipeline().device().create_buffer<float>(pixel_count * 3u);

        CounterBuffer accept_counter;
        CounterBuffer mutation_counter;
        CounterBuffer global_accept_counter;
        Shader1D<> clear_statistics;
        if (node<PSSMLT>()->enable_statistics()) {
            accept_counter = {pipeline().device(), pixel_count};
            mutation_counter = {pipeline().device(), pixel_count};
            global_accept_counter = {pipeline().device(), 1u};
            clear_statistics = pipeline().device().compile<1u>([&] {
                auto i = dispatch_x();
                $if(i == 0u) { global_accept_counter.clear(i); };
                accept_counter.clear(i);
                mutation_counter.clear(i);
            });
            command_buffer << clear_statistics().dispatch(pixel_count);
        }

        auto rng_state_buffer = pipeline().device().create_buffer<uint>(chains);

        Clock clk;
        LUISA_INFO("PSSMLT: compiling create_chains kernel...");
        auto create_chains = pipeline().device().compile<1u>([&](Float time, Float shutter_weight) noexcept {
            auto chain_id = dispatch_x();
            auto u_bootstrap = uniform_uint_to_float(xxhash32(make_uint2(chain_id, 0x19980810u)));
            auto [bootstrap_id, _] = sample_alias_table(bootstrap_sampling_table,
                                                        static_cast<uint>(bootstrap_sampling_table.size()),
                                                        u_bootstrap);
            _sampler->create(chain_id, bootstrap_id);
            auto seed = xxhash32(make_uint2(bootstrap_id, 0xdeadbeefu));
            auto [p, L, is_light] = Li(*_sampler, seed, camera, time);
            position_buffer->write(chain_id, p);
            radiance_and_contribution_buffer->write(
                chain_id, make_float4(L * shutter_weight, _s(L, is_light)));
            rng_state_buffer->write(chain_id, seed);
            _sampler->save();
        });
        LUISA_INFO("PSSMLT: compiled create_chains kernel in {} ms.", clk.toc());

        clk.tic();
        LUISA_INFO("PSSMLT: compiling render kernel...");
        auto propose = pipeline().device().compile<1u>([&](Float time, Float shutter_weight, Float b) noexcept {
            auto chain_id = dispatch_id().x;
            auto u_wavelength = def(0.f);
            auto seed = rng_state_buffer->read(chain_id);
            _sampler->load(chain_id);
            _sampler->start_iteration();
            auto proposed = Li(*_sampler, seed, camera, time);
            auto p_new = std::get<0u>(proposed);
            auto L_new = std::get<1u>(proposed);
            auto y_new = _s(L_new, std::get<2u>(proposed));
            auto p_old = position_buffer->read(chain_id);
            auto L_and_y_old = radiance_and_contribution_buffer->read(chain_id);
            auto L_old = L_and_y_old.xyz();
            auto y_old = L_and_y_old.w;

            auto accum = [&accumulate_buffer, resolution](Expr<uint2> p, Expr<float3> L) noexcept {
                auto offset = (p.y * resolution.x + p.x) * 3u;
                $if(!any(isnan(L))) {
                    for (auto i = 0u; i < 3u; i++) {
                        accumulate_buffer->atomic(offset + i).fetch_add(L[i]);
                    }
                };
            };

            // Compute acceptance probability for proposed sample
            auto accept = clamp(y_new / y_old, 0.f, 1.f);
            // Splat both current and proposed samples to _film_
            // Use the MIS weights proposed by [Csaba Kelemen and László Szirmay-Kalos, 2001]
            auto w_new = (accept + ite(_sampler->large_step(), 1.f, 0.f)) / (y_new / b + p_large);
            auto w_old = (1.f - accept) / (y_old / b + p_large);
            accum(p_new, shutter_weight * w_new * L_new);
            accum(p_old, w_old * L_old);
            auto pixel_index_new = p_new.x + p_new.y * resolution.x;
            mutation_counter.record(pixel_index_new);

            // Accept or reject the proposal
            $if(lcg(seed) < accept) {
                position_buffer->write(chain_id, p_new);
                radiance_and_contribution_buffer->write(
                    chain_id, make_float4(shutter_weight * L_new, y_new));
                _sampler->accept();
                accept_counter.record(pixel_index_new);
                global_accept_counter.record(0u);
            }
            $else {
                _sampler->reject();
            };
            rng_state_buffer->write(chain_id, seed);
            _sampler->save();
        });
        LUISA_INFO("PSSMLT: compiled render kernel in {} ms.", clk.toc());

        clk.tic();
        LUISA_INFO("PSSMLT: compiling clear kernel...");
        auto clear = pipeline().device().compile<1u>([&]() noexcept {
            auto pixel_id = dispatch_id().x;
            accumulate_buffer->write(pixel_id * 3u + 0u, 0.f);
            accumulate_buffer->write(pixel_id * 3u + 1u, 0.f);
            accumulate_buffer->write(pixel_id * 3u + 2u, 0.f);
        });
        LUISA_INFO("PSSMLT: compiled clear kernel in {} ms.", clk.toc());

        clk.tic();
        LUISA_INFO("PSSMLT: compiling accumulate kernel...");
        auto accumulate = pipeline().device().compile<2>([&](Float effective_spp) {
            auto p = dispatch_id().xy();
            auto offset = (p.y * resolution.x + p.x) * 3u;
            auto L = make_float3(accumulate_buffer->read(offset + 0u),
                                 accumulate_buffer->read(offset + 1u),
                                 accumulate_buffer->read(offset + 2u));
            camera->film()->accumulate(p, L, effective_spp);
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
        auto total_mutations = static_cast<uint64_t>(camera->node()->spp()) * pixel_count;
        auto last_effective_spp = 0.;
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            auto mutations = static_cast<uint64_t>(s.spp) * pixel_count;
            auto mutations_per_chain = (mutations + chains - 1u) / chains;
            for (auto i = static_cast<uint64_t>(0u); i < mutations_per_chain; i++) {
                auto chains_to_dispatch = std::min((i + 1u) * chains, mutations) - i * chains;
                command_buffer << propose(s.point.time, s.point.weight, static_cast<float>(b))
                                      .dispatch(chains_to_dispatch);
                mutation_count += chains_to_dispatch;
                dispatch_count++;
                if (camera->film()->show(command_buffer)) { dispatch_count = 0u; }
                auto dispatches_per_commit = 16u;
                if (dispatch_count >= dispatches_per_commit) [[unlikely]] {
                    auto p = static_cast<double>(mutation_count) /
                             static_cast<double>(total_mutations);
                    auto effective_spp = p * camera->node()->spp();
                    command_buffer << accumulate(static_cast<float>(effective_spp - last_effective_spp))
                                          .dispatch(resolution)
                                   << clear().dispatch(pixel_count);
                    last_effective_spp = effective_spp;
                    if (node<PSSMLT>()->enable_statistics()) {
                        auto a = luisa::make_shared<uint64_t>();
                        command_buffer << global_accept_counter.copy_to(a.get())
                                       << [a, total = mutation_count] {
                                              auto accepted = *a;
                                              auto rate = static_cast<double>(accepted) / static_cast<double>(total);
                                              LUISA_INFO("PSSMLT: {}/{} mutation(s) accepted ({:.2f}%).",
                                                         accepted, total, rate * 100.);
                                          };
                    }
                    dispatch_count = 0u;
                    command_buffer << [&progress, p] { progress.update(p); };
                }
            }
        }
        // final
        command_buffer << accumulate(static_cast<float>(camera->node()->spp() - last_effective_spp))
                              .dispatch(resolution)
                       << synchronize();
        progress.done();
        auto render_time = clk.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);

        // retrieve statistics
        if (node<PSSMLT>()->enable_statistics()) {
            LUISA_INFO("PSSMLT: saving statistic images...");
            luisa::vector<uint64_t> accept(pixel_count);
            luisa::vector<uint64_t> mutation(pixel_count);
            command_buffer << accept_counter.copy_to(accept.data())
                           << mutation_counter.copy_to(mutation.data())
                           << synchronize();
            luisa::vector<float> accept_rate(pixel_count);
            luisa::vector<float> density(pixel_count);
            std::transform(accept.cbegin(), accept.cend(), mutation.cbegin(), accept_rate.begin(),
                           [](auto a, auto m) noexcept {
                               return static_cast<float>(static_cast<double>(a) /
                                                         static_cast<double>(m));
                           });
            std::transform(mutation.cbegin(), mutation.cend(), density.begin(),
                           [n = static_cast<double>(camera->node()->spp())](auto m) noexcept {
                               return static_cast<float>(static_cast<double>(m) / n);
                           });
            auto rate_file_name = camera->node()->file();
            auto density_file_name = camera->node()->file();
            rate_file_name.replace_extension(".accept.exr");
            density_file_name.replace_extension(".density.exr");
            save_image(rate_file_name, accept_rate.data(), resolution, 1u);
            save_image(density_file_name, density.data(), resolution, 1u);
        }
    }

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        auto shutter_samples = camera->node()->shutter_samples();

        // reset sampler
        _sampler->reset(command_buffer, node<PSSMLT>()->chains(),
                        _compute_pss_dimension(camera->node()));

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
