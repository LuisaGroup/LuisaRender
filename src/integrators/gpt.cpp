#include "EASTL/unique_ptr.h"
#include "base/geometry.h"
#include "base/interaction.h"
#include "base/sampler.h"
#include "base/spectrum.h"
#include "core/basic_traits.h"
#include "core/basic_types.h"
#include "core/mathematics.h"
#include "core/stl.h"
#include "dsl/builtin.h"
#include "dsl/expr.h"
#include "dsl/sugar.h"
#include "dsl/var.h"
#include "rtx/ray.h"
#include "util/frame.h"
#include "util/scattering.h"
#include "util/spec.h"
#include <cmath>
#include <limits>
#include <util/imageio.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <util/sampling.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/scene.h>

namespace luisa::render {

using namespace compute;

static const luisa::unordered_map<luisa::string_view, uint>
    aov_component_to_channels{{"sample", 3u},
                              {"diffuse", 3u},
                              {"specular", 3u},
                              {"normal", 3u},
                              {"albedo", 3u},
                              {"depth", 1u},
                              {"roughness", 2u},
                              {"ndc", 3u},
                              {"mask", 1u}};

class GradientPathTracing final : public Integrator {
private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;

public:
    GradientPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Integrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class GradientPathTracingInstance final : public Integrator::Instance {

private:
    uint _last_spp{0u};
    Clock _clock;
    Framerate _framerate;
    luisa::optional<Window> _window;

private:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept;

public:
    explicit GradientPathTracingInstance(
        const GradientPathTracing *node,
        Pipeline &pipeline, CommandBuffer &cmd_buffer) noexcept
        : Integrator::Instance{pipeline, cmd_buffer, node} {
    }

    void render(Stream &stream) noexcept override {
        auto pt = node<GradientPathTracing>();
        auto command_buffer = stream.command_buffer();
        for (auto i = 0u; i < pipeline().camera_count(); i++) {
            auto camera = pipeline().camera(i);
            auto resolution = camera->film()->node()->resolution();
            auto pixel_count = resolution.x * resolution.y;
            _last_spp = 0u;
            _clock.tic();
            _framerate.clear();
            camera->film()->prepare(command_buffer);
            _render_one_camera(command_buffer, camera);
            command_buffer << compute::synchronize();
            camera->film()->release();
        }
        while (_window && !_window->should_close()) {
            _window->run_one_frame([] {});
        }
    }
};

luisa::unique_ptr<Integrator::Instance> GradientPathTracing::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<GradientPathTracingInstance>(
        this, pipeline, command_buffer);
}

const float D_EPSILON = 1e-14f;

struct GPTConfig {
    uint m_max_depth;
	uint m_min_depth;
    uint m_rr_depth;
	bool m_strict_normals;
	float m_shift_threshold;
	bool m_reconstruct_L1;
	bool m_reconstruct_L2;
	float m_reconstruct_alpha;
};

enum VertexType : uint {
    VERTEX_TYPE_GLOSSY,
    VERTEX_TYPE_DIFFUSE
};

enum RayConnection : uint {
	RAY_NOT_CONNECTED,
	RAY_RECENTLY_CONNECTED,
	RAY_CONNECTED
};

struct RayState {
    RayDifferential ray;
    SampledSpectrum throughput;
    Float pdf;
    SampledSpectrum radiance;
    SampledSpectrum gradient;
    // RadianceQueryRecord rRec;        ///< The radiance query record for this ray.
    luisa::shared_ptr<Interaction> it;
    Float eta;                      // Current refractive index
    Bool alive;
    UInt connection_status;

    RayState() : radiance(0.0f), gradient(0.0f), it(nullptr), eta(1.0f), pdf(1.0f), 
                 throughput(0.0f), alive(true), connection_status((uint) RAY_NOT_CONNECTED) {}

    inline void add_radiance(const SampledSpectrum& contribution, Expr<float> weight) noexcept {
        auto color = contribution * weight;
        radiance += color;
    }

    inline void add_gradient(const SampledSpectrum& contribution, Expr<float> weight) noexcept {
        auto color = contribution * weight;
        gradient += color;
    }
};

const float epsilon = 1e-4f;
const float shadow_epsilon = 1e-3f;

[[nodiscard]] auto test_visibility(const Pipeline* pipeline, Expr<float3> point1, Expr<float3> point2) noexcept {
    auto shadow_ray = make_ray(
        point1, point2 - point1, epsilon, 1.f - shadow_epsilon
    );
    return !pipeline->geometry()->intersect_any(shadow_ray);
}

[[nodiscard]] auto test_environment_visibility(const Pipeline* pipeline, const Var<Ray>& ray) noexcept {
    if(!pipeline->environment()) return def(false);
    auto shadow_ray = make_ray(
        ray->origin(), ray->direction(), epsilon, std::numeric_limits<float>::max()
    );
    // Miss = Intersect with env TODO
    return !pipeline->geometry()->intersect_any(shadow_ray); 
}

[[nodiscard]] auto get_vertex_type_by_roughness(Expr<float> roughness, const GPTConfig* config) noexcept {
    return ite(roughness <= config->m_shift_threshold, (uint) VertexType::VERTEX_TYPE_GLOSSY, (uint)VertexType::VERTEX_TYPE_DIFFUSE);
}

[[nodiscard]] auto get_vertex_type(const Pipeline* pipeline, luisa::shared_ptr<Interaction> it, const GPTConfig* config, const SampledWavelengths& swl, Expr<float> time) noexcept {
    auto surface_tag = it->shape()->surface_tag();
    Float2 roughness;
    pipeline->surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
        auto closure = surface->closure(it, swl, 1.f, time);
        roughness = closure->roughness();
    });
    return get_vertex_type_by_roughness(min(roughness.x, roughness.y), config);
}

struct HalfVectorShiftResult {
    Bool success;
    Float jacobian;
    Float3 wo;
};

[[nodiscard]] HalfVectorShiftResult half_vector_shift(
    Float3 tangent_space_main_wi,
    Float3 tangent_space_main_wo,
    Float3 tangent_space_shifted_wi,
    Float main_eta, Float shifted_eta) noexcept {
    HalfVectorShiftResult result;

    $if(cos_theta(tangent_space_main_wi) * cos_theta(tangent_space_shifted_wi) < 0.f) {
        // Refraction

        $if(main_eta == 1.f | shifted_eta == 1.f) {
            result.success = false;
        } $else {
            auto tangent_space_half_vector_non_normalized_main = ite(
                cos_theta(tangent_space_main_wi) < 0.f,
                -(tangent_space_main_wi * main_eta + tangent_space_main_wo),
                -(tangent_space_main_wi + tangent_space_main_wo * main_eta)
            );

            auto tangent_space_half_vector = normalize(tangent_space_half_vector_non_normalized_main);
            
            Float3 tangent_space_shifted_wo; 
            auto refract_not_internal = refract(tangent_space_shifted_wi, tangent_space_half_vector, shifted_eta, &tangent_space_shifted_wo); 

            $if(!refract_not_internal) {
                result.success = false;
            } $else {
                auto tangent_space_half_vector_non_normalized_shifted = ite(
                    cos_theta(tangent_space_shifted_wi) < 0.f,
                    -(tangent_space_shifted_wi * shifted_eta + tangent_space_shifted_wo),
                    -(tangent_space_shifted_wi + tangent_space_shifted_wo * shifted_eta)
                );

                auto h_length_squared = length_squared(tangent_space_half_vector_non_normalized_shifted);
                auto wo_dot_h = abs(dot(tangent_space_main_wo, tangent_space_half_vector)) / (D_EPSILON + abs(dot(tangent_space_shifted_wo, tangent_space_half_vector)));
                
                result.success = true;
                result.wo = tangent_space_shifted_wo;
                result.jacobian = h_length_squared * wo_dot_h;
            };
        };
    } $else {
        // Reflection
        auto tangent_space_half_vector = normalize(tangent_space_main_wi + tangent_space_main_wo);
        auto tangent_space_shifted_wo = reflect(tangent_space_shifted_wi, tangent_space_half_vector);

        auto wo_dot_h = dot(tangent_space_shifted_wo, tangent_space_half_vector) / dot(tangent_space_main_wo, tangent_space_half_vector);
        auto jacobian = abs(wo_dot_h);

        result.success = true;
        result.wo = tangent_space_shifted_wo;
        result.jacobian = jacobian;
    };

    return result;
}

struct ReconnectionShiftResult {
    Bool success;
    Float jacobian;
    Float3 wo;
};

[[nodiscard]] ReconnectionShiftResult reconnect_shift(
    const Pipeline* pipeline, 
    Expr<float3> main_source_vertex, 
    Expr<float3> target_vertex,
    Expr<float3> shift_source_vertex,
    Expr<float3> target_normal) noexcept {
    ReconnectionShiftResult result;
    result.success = false;
    $if(test_visibility(pipeline, shift_source_vertex, target_vertex)) {
        auto main_edge = main_source_vertex - target_vertex;
        auto shifted_edge = shift_source_vertex - target_vertex;

        auto main_edge_length_squared = length_squared(main_edge);
        auto shifted_edge_length_squared = length_squared(shifted_edge);

        auto shifted_wo = -shifted_edge / sqrt(shifted_edge_length_squared);

        auto main_opposing_cosine = dot(main_edge, target_normal) / sqrt(main_edge_length_squared);
        auto shifted_opposing_cosine = dot(shifted_wo, target_normal);

        auto jacobian = abs(shifted_opposing_cosine * main_edge_length_squared) / (D_EPSILON + abs(main_opposing_cosine * shifted_edge_length_squared));

        result.success = true;
        result.jacobian = jacobian;
        result.wo = shifted_wo;
    };

    return result;
}

[[nodiscard]] ReconnectionShiftResult environment_shift(
    const Pipeline* pipeline,
    const Var<Ray>& main_ray,
    Expr<float3> shift_source_vertex) noexcept {
    ReconnectionShiftResult result;
    result.success = false;

    auto offset_ray = make_ray(
        shift_source_vertex, main_ray->direction(), main_ray->t_min(), main_ray->t_max()
    );

    $if(test_environment_visibility(pipeline, offset_ray)) {
        result.success = true;
        result.jacobian = 1.f;
        result.wo = main_ray->direction();
    };

    return result;
}

struct SurfaceSampleResult {
    Surface::Sample sample;
    SampledSpectrum weight;
    Float pdf;
    Float3 wo;
    Float eta;
};

class GradientPathTracingImpl {
private:
    const Pipeline* _pipeline;
    Sampler::Instance* _sampler;
    LightSampler::Instance* _light_sampler;
    const Camera::Instance* _camera;
    const GPTConfig* _config;

public:
    struct Evaluation {
        SampledSpectrum very_direct;
        SampledSpectrum throughput;
        SampledSpectrum gradients[4];
        SampledSpectrum neighbor_throughput[4];
        SampledWavelengths swl;
    };

    GradientPathTracingImpl(const Pipeline* pipeline, Sampler::Instance* sampler, LightSampler::Instance* light_sampler, const Camera::Instance* camera, const GPTConfig* config)
        : _pipeline(pipeline), _sampler(sampler),_light_sampler(light_sampler) ,_camera(camera), _config(config) {}

    // Entrance function of GPT
    [[nodiscard]] auto evaluate_point(Expr<uint2> pixel_coord, Expr<uint> sample_index, Expr<float> time, float diff_scale_factor) noexcept {
        _sampler->start(pixel_coord, sample_index);
        auto u_filter = _sampler->generate_pixel_2d();
        auto u_lens = _camera->node()->requires_lens_sampling() ? _sampler->generate_2d() : make_float2(.5f);
        auto [main_ray_diff, _, main_ray_weight] = _camera->generate_ray_differential(pixel_coord, time, u_filter, u_lens);
        auto spectrum = _pipeline->spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : _sampler->generate_1d());

        RayState main_ray;
        main_ray.ray = main_ray_diff;
        main_ray.ray.scale_differential(diff_scale_factor);
        main_ray.throughput = SampledSpectrum{swl.dimension(), main_ray_weight};
        // TODO: rRec

        RayState shifted_rays[4];
        static const UInt2 pixel_shifts[4] = {
            make_uint2(1, 0),
            make_uint2(0, 1),
            make_uint2(-1, 0),
            make_uint2(0, -1)
        };

        for(int i = 0; i < 4; i++) {
            auto [shifted_diff, _, shifted_weight] = _camera->generate_ray_differential(pixel_coord + pixel_shifts[i], time, u_filter, u_lens);
            shifted_rays[i].ray = shifted_diff;
            shifted_rays[i].ray.scale_differential(diff_scale_factor);
            shifted_rays[i].throughput = SampledSpectrum{swl.dimension(), shifted_weight};
            // TODO: rRec
        }

        // Actual implementation
        SampledSpectrum very_direct = evaluate(main_ray, shifted_rays, swl, time);

        return Evaluation{
            .very_direct = very_direct,
            .throughput = main_ray.radiance,
            .gradients = {
                shifted_rays[0].gradient,
                shifted_rays[1].gradient,
                shifted_rays[2].gradient,
                shifted_rays[3].gradient
            },
            .neighbor_throughput = {
                shifted_rays[0].radiance,
                shifted_rays[1].radiance,
                shifted_rays[2].radiance,
                shifted_rays[3].radiance
            },
            .swl = swl
        };
    }

    [[nodiscard]] SurfaceSampleResult sample_surface(RayState& state, SampledWavelengths& swl, Expr<float> time) noexcept {
        auto& it = state.it;
        auto& ray = state.ray;

        SurfaceSampleResult result {
            .sample = Surface::Sample::zero(swl.dimension()),
            .weight = SampledSpectrum{swl.dimension(), 0.f},
            .pdf = def(0.f),
            .eta = def(0.f)
        };
        auto surface_tag = it->shape()->surface_tag();
        auto u_lobe = _sampler->generate_1d();
        auto u_bsdf = _sampler->generate_2d();
        _pipeline->surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
            auto closure = surface->closure(it, swl, 1.f, time);
            result.sample = closure->sample(-state.ray.ray->direction(), u_lobe, u_bsdf);
            result.eta = closure->eta().value_or(1.f);
        });
        result.weight = result.sample.eval.f;
        result.pdf = result.sample.eval.pdf;
        result.wo = -state.ray.ray->direction();

        return result;
    }

    [[nodiscard]] SampledSpectrum evaluate(RayState& main, RayState* shifteds, SampledWavelengths& swl, Expr<float> time) noexcept {
        SampledSpectrum result{swl.dimension(), 0.f};

        main.it = _pipeline->geometry()->intersect(main.ray.ray);
        main.ray.ray->set_t_min(epsilon);

        for(int i = 0; i < 4; i++) {
            auto& shifted =shifteds[i];
            shifted.it = _pipeline->geometry()->intersect(shifted.ray.ray);
            shifted.ray.ray->set_t_min(epsilon);
        }

        $if(!main.it->valid()) {
            if(_pipeline->environment()) {
                auto eval = _light_sampler->evaluate_miss(main.ray.ray->direction(), swl, time);
                result += main.throughput * eval.L;
            };
        } $else {
            if(!_pipeline->lights().empty()) {
                $if(main.it->shape()->has_light()) {
                    auto eval = _light_sampler->evaluate_hit(*main.it, main.ray.ray->origin(), swl, time);
                    result += main.throughput * eval.L;
                };
            }
            // Subsurface omitted TODO

            for(int i = 0; i < 4; i++) {
                auto& shifted = shifteds[i];
                $if(!shifted.it->valid()) {
                    shifted.alive = false;
                };
            }

            // Strict normal check to produce the same results as bidirectional methods when normal mapping is used. 
            // TODO

            // Main PT Loop
            $for(depth, _config->m_max_depth) {
                // Strict normal check to produce the same results as bidirectional methods when normal mapping is used. 
                // TODO

                auto last_segment = depth + 1 == _config->m_max_depth;

                //
                // Direct Illumination Sampling
                //

                // Sample incoming radiance from lights (next event estimation)
                auto u_light_selection = _sampler->generate_1d();
                auto u_light_surface = _sampler->generate_2d();
                auto main_light_sample = _light_sampler->sample(*main.it, u_light_selection, u_light_surface, swl, time);
                // TODO: main.it = main_light_sample?
                auto main_occluded_it = _pipeline->geometry()->intersect(main_light_sample.shadow_ray);

                auto main_surface_tag = main.it->shape()->surface_tag();
                auto wo = -main.ray.ray->direction();
                $if(main_light_sample.eval.pdf > 0.f & !main_occluded_it->valid()) {
                    auto wi = main_light_sample.shadow_ray->direction();

                    Surface::Evaluation main_light_eval{.f = SampledSpectrum{0.f}, .pdf=0.f};
                    _pipeline->surfaces().dispatch(main_surface_tag, [&](auto surface) noexcept {
                        auto closure = surface->closure(main.it, swl, 1.f, time);
                        main_light_eval = closure->evaluate(wo, wi);
                    });

                    auto main_distance_squared = length_squared(main.it->p() - main_occluded_it->p());
                    auto main_opposing_cosine = dot(main_occluded_it->ng(), main.it->p() - main_occluded_it->p()) / sqrt(main_distance_squared);

                    auto main_weight_numerator = main.pdf * main_light_sample.eval.pdf;
                    auto main_bsdf_pdf = main_light_eval.pdf;
                    auto main_weight_denominator = main.pdf * main.pdf  * (
                        main_light_sample.eval.pdf * main_light_sample.eval.pdf + main_bsdf_pdf * main_bsdf_pdf
                    );

                    if(!_config->m_strict_normals) { // strict normal not implemented TODO
                        for(int i = 0; i < 4; i++) {
                            auto& shifted = shifteds[i];
                            SampledSpectrum main_contribution{ swl.dimension(), 0.f };
                            SampledSpectrum shifted_contribution{ swl.dimension(), 0.f };
                            auto weight = def(0.f);

                            auto shift_successful = shifted.alive;

                            $if(shift_successful) {
                                $switch(shifted.connection_status) {
                                    $case((uint) RayConnection::RAY_CONNECTED) {
                                        auto shifted_bsdf_pdf = main_bsdf_pdf;
                                        auto shifted_emitter_pdf = main_light_sample.eval.pdf;
                                        auto shifted_bsdf_value = main_light_eval.f;
                                        auto shifted_emitter_radiance = main_light_sample.eval.L * main_light_sample.eval.pdf;
                                        auto jacobian = 1.f;

                                        auto shifted_weight_denominator = (jacobian * shifted.pdf) * (jacobian * shifted.pdf) * (
                                            shifted_emitter_pdf * shifted_emitter_pdf + shifted_bsdf_pdf * shifted_bsdf_pdf
                                        );
                                        weight = main_weight_numerator / (D_EPSILON + shifted_weight_denominator + main_weight_denominator);
                                        main_contribution = main.throughput * main_light_eval.f * main_light_sample.eval.L * main_light_sample.eval.pdf;
                                        shifted_contribution = jacobian * shifted.throughput * (shifted_bsdf_value * shifted_emitter_radiance);
                                    };
                                    $case((uint) RayConnection::RAY_RECENTLY_CONNECTED) {
                                        auto incoming_direction = normalize(shifted.it->p() - main.it->p());

                                        Surface::Evaluation shifted_bsdf_eval{ .f=SampledSpectrum{0.f}, .pdf=0.f };
                                        _pipeline->surfaces().dispatch(main_surface_tag, [&](auto surface) noexcept {
                                            auto closure = surface->closure(main.it, swl, 1.f, time);
                                            shifted_bsdf_eval = closure->evaluate(-incoming_direction, main_light_sample.shadow_ray->direction());
                                        });
                                        auto shifted_emitter_pdf = main_light_sample.eval.pdf;
                                        auto shifted_bsdf_value = shifted_bsdf_eval.f;
                                        auto shifted_bsdf_pdf = ite(main_occluded_it->valid(), shifted_bsdf_eval.pdf, 0.f);
                                        auto shifted_emitter_radiance = main_light_sample.eval.L * main_light_sample.eval.pdf;
                                        auto jacobian = 1.f;

                                        auto shifted_weight_denominator = (jacobian * shifted.pdf) * (jacobian * shifted.pdf) * (
                                            shifted_emitter_pdf * shifted_emitter_pdf + shifted_bsdf_pdf * shifted_bsdf_pdf
                                        );
                                        weight = main_weight_numerator / (D_EPSILON + shifted_weight_denominator + main_weight_denominator);
                                        main_contribution = main.throughput * main_light_eval.f * main_light_sample.eval.L * main_light_sample.eval.pdf;
                                        shifted_contribution = jacobian * shifted.throughput * (shifted_bsdf_value * shifted_emitter_radiance);
                                    };
                                    $case((uint) RayConnection::RAY_NOT_CONNECTED) {
                                        // TODO
                                        auto main_vertex_type = get_vertex_type(_pipeline, main.it, _config, swl, time);
                                        auto shifted_vertex_type = get_vertex_type(_pipeline, shifted.it, _config, swl, time);

                                        $if(main_vertex_type == (uint) VertexType::VERTEX_TYPE_DIFFUSE & shifted_vertex_type == (uint) VertexType::VERTEX_TYPE_DIFFUSE) {
                                            auto shifted_light_sample = _light_sampler->sample(*shifted.it, u_light_selection, u_light_surface, swl, time);
                                            // TODO: shifted.it = shifted_light_sample?
                                            auto shifted_occluded_it = _pipeline->geometry()->intersect(shifted_light_sample.shadow_ray);

                                            auto shifted_emitter_radiance = shifted_light_sample.eval.L * shifted_light_sample.eval.pdf;
                                            auto shifted_drec_pdf = shifted_light_sample.eval.pdf;
                                            
                                            auto shifted_distance_squared = length_squared(shifted.it->p() - shifted_occluded_it->p());
                                            auto emitter_direction = (shifted.it->p() - shifted_occluded_it->p()) / sqrt(shifted_distance_squared);
                                            auto shifted_opposing_cosine = -dot(shifted_occluded_it->ng(), emitter_direction);

                                            // TODO: No strict normal here
                                            auto shifted_surface_tag = shifted.it->shape()->surface_tag();
                                            Surface::Evaluation shifted_light_eval{.f = SampledSpectrum{0.f}, .pdf=0.f};
                                            _pipeline->surfaces().dispatch(shifted_surface_tag, [&](auto surface) noexcept {
                                                auto closure = surface->closure(shifted.it, swl, 1.f, time);
                                                shifted_light_eval = closure->evaluate(-shifted.ray.ray->direction(), -emitter_direction); // TODO: check +-
                                            });

                                            auto shifted_bsdf_value = shifted_light_eval.f;
                                            auto shifted_bsdf_pdf = ite(shifted_occluded_it->valid(), shifted_light_eval.pdf, 0.f);
                                            auto jacobian = abs(shifted_opposing_cosine * main_distance_squared) / (epsilon + abs(main_opposing_cosine * shifted_distance_squared));

                                            auto shifted_weight_denominator = (jacobian * shifted.pdf) * (jacobian * shifted.pdf) * (
                                                shifted_drec_pdf * shifted_drec_pdf + shifted_bsdf_pdf * shifted_bsdf_pdf
                                            );
                                            weight = main_weight_numerator / (D_EPSILON + shifted_weight_denominator + main_weight_denominator);

                                            main_contribution = main.throughput * main_light_eval.f * main_light_sample.eval.L * main_light_sample.eval.pdf;
                                            shifted_contribution = jacobian * shifted.throughput * (shifted_bsdf_value * shifted_emitter_radiance);
                                        };
                                    };
                                };
                            } $else { // shift_successful == false
                                auto shifted_weight_denominator = 0.f;
                                weight = main_weight_numerator / (D_EPSILON + main_weight_denominator);

                                main_contribution = main.throughput * main_light_eval.f * main_light_sample.eval.L * main_light_sample.eval.pdf;
                                shifted_contribution = SampledSpectrum{ swl.dimension(), 0.f };
                            };

                            // TODO: Check if add valid
                            main.add_radiance(main_contribution, weight);
                            shifted.add_radiance(shifted_contribution, weight);

                            shifted.add_gradient(shifted_contribution - main_contribution, weight);
                        }
                    }
                };

                //
                // BSDF Sampling & Emitter
                //

                auto main_bsdf_result = sample_surface(main, swl, time);
                $if(main_bsdf_result.pdf > 0.f) {
                    auto main_wo = main.ray.ray->direction();

                    auto main_wo_dot_ng = dot(main.it->ng(), main_wo);
                    // TODO: strict normal

                    auto previous_main_it = *main.it;

                    auto main_hit_emitter = def(false);
                    auto main_emitter_radiance = SampledSpectrum{swl.dimension(), 0.f};
                    auto main_emitter_pdf = def(0.f);

                    auto main_vertex_type = get_vertex_type(_pipeline, main.it, _config, swl, time);
                    auto main_next_vertex_type = def(0u);
                    auto neither_hit_nor_env_flag = def(false);

                    main.ray = RayDifferential{ .ray = make_ray(main.it->p(), main_wo) };
                    main.it = _pipeline->geometry()->intersect(main.ray.ray);
                    $if(main.it->valid()) {
                        if(!_pipeline->lights().empty()) {
                            $if(main.it->shape()->has_light()) {
                                auto eval = _light_sampler->evaluate_hit(*main.it, main.ray.ray->origin(), swl, time);
                                main_emitter_radiance = eval.L;
                                main_emitter_pdf = eval.pdf;
                                main_hit_emitter = true;
                                // TODO: setQuery?
                            };
                        }

                        // TODO: subsurface scattering

                        main_next_vertex_type = get_vertex_type(_pipeline, main.it, _config, swl, time);
                    } $else {
                        if(_pipeline->environment()) {
                            auto eval = _light_sampler->evaluate_miss(main.ray.ray->direction(), swl, time);
                            main_emitter_radiance = eval.L;
                            main_emitter_pdf = eval.pdf;
                            main_hit_emitter = true;

                            main_next_vertex_type = (uint) VERTEX_TYPE_DIFFUSE;
                        } else {
                            neither_hit_nor_env_flag = true;
                        }
                    };

                    $if(!neither_hit_nor_env_flag) { // hit something or hit environment
                        // Continue the shift
                        auto main_bsdf_pdf = main_bsdf_result.pdf;
                        auto main_previous_pdf = main.pdf;

                        main.throughput *= main_bsdf_result.weight * main_bsdf_result.pdf;
                        main.pdf *= main_bsdf_result.pdf;
                        main.eta *= main_bsdf_result.eta;

                        auto main_lum_pdf = ite(
                            main_hit_emitter & depth + 1u >= _config->m_max_depth /* TODO :& !(mainBsdfResult.bRec.sampledType & BSDF::EDelta)*/,
                            main_emitter_pdf, 0.f
                        );

                        auto main_weight_numerator = main_previous_pdf * main_bsdf_result.pdf;
                        auto main_weight_denominator = (main_previous_pdf * main_previous_pdf) * (
                            main_lum_pdf * main_lum_pdf + main_bsdf_pdf * main_bsdf_pdf
                        );

                        for(int i = 0; i < 4; i++) {
                            auto& shifted = shifteds[i];

                            SampledSpectrum shifted_emitter_radiance{ swl.dimension(), 0.f };
                            SampledSpectrum main_contribution{ swl.dimension(), 0.f };
                            SampledSpectrum shifted_contribution{ swl.dimension(), 0.f };
                            auto weight = def(0.f);

                            auto postponed_shift_end = def(false); // Kills the shift after evaluating the current radiance.

                            $if(shifted.alive) {
                                auto shifted_previous_pdf = shifted.pdf;
                                $switch(shifted.connection_status) {
                                    $case((uint) RayConnection::RAY_CONNECTED) {
                                        auto shifted_bsdf_value = main_bsdf_result.weight * main_bsdf_result.pdf;
                                        auto shifted_bsdf_pdf = main_bsdf_pdf;
                                        auto shifted_lum_pdf = main_lum_pdf;
                                        auto shifted_emitter_radiance = main_emitter_radiance;

                                        shifted.throughput *= shifted_bsdf_value;
                                        shifted.pdf *= shifted_bsdf_pdf;

                                        auto shifted_weight_denominator = (shifted_previous_pdf * shifted_previous_pdf) * (
                                            shifted_lum_pdf * shifted_lum_pdf + shifted_bsdf_pdf * shifted_bsdf_pdf
                                        );

                                        weight = main_weight_numerator / (D_EPSILON + shifted_weight_denominator + main_weight_denominator);

                                        main_contribution = main.throughput * main_emitter_radiance;
                                        shifted_contribution = shifted.throughput * shifted_emitter_radiance;
                                    };
                                    $case((uint) RayConnection::RAY_RECENTLY_CONNECTED) {
                                        auto incoming_direction = normalize(shifted.it->p() - main.ray.ray->origin());
                                        Surface::Evaluation shifted_bsdf_eval{ .f=SampledSpectrum{0.f}, .pdf=0.f };
                                        _pipeline->surfaces().dispatch(previous_main_it.shape()->surface_tag(), [&](auto surface) noexcept {
                                            auto closure = surface->closure(make_shared<Interaction>(previous_main_it), swl, 1.f, time);
                                            shifted_bsdf_eval = closure->evaluate(-incoming_direction, main_light_sample.shadow_ray->direction());
                                        });

                                        auto shifted_bsdf_value = shifted_bsdf_eval.f;
                                        auto shifted_bsdf_pdf = shifted_bsdf_eval.pdf;
                                        auto shifted_lum_pdf = main_lum_pdf;
                                        auto shifted_emitter_radiance = main_emitter_radiance;

                                        shifted.throughput *= shifted_bsdf_value;
                                        shifted.pdf *= shifted_bsdf_pdf;

                                        shifted.connection_status = (uint) RayConnection::RAY_CONNECTED;

                                        auto shifted_weight_denominator = (shifted_previous_pdf * shifted_previous_pdf) * (
                                            shifted_lum_pdf * shifted_lum_pdf + shifted_bsdf_pdf * shifted_bsdf_pdf
                                        );

                                        weight = main_weight_numerator / (D_EPSILON + shifted_weight_denominator + main_weight_denominator);

                                        main_contribution = main.throughput * main_emitter_radiance;
                                        shifted_contribution = shifted.throughput * shifted_emitter_radiance;
                                    };
                                    $case((uint) RayConnection::RAY_NOT_CONNECTED) {
                                        auto shifted_vertex_type = get_vertex_type(_pipeline, shifted.it, _config, swl, time);
                                        $if(main_vertex_type == (uint) VertexType::VERTEX_TYPE_DIFFUSE & main_next_vertex_type == (uint) VertexType::VERTEX_TYPE_DIFFUSE 
                                            & shifted_vertex_type == (uint) VertexType::VERTEX_TYPE_DIFFUSE) {
                                            // Reconnect shift
                                            $if(!last_segment | main_hit_emitter /*| main.it has subsurface TODO*/) {
                                                ReconnectionShiftResult shift_result;
                                                auto environment_connection = def(false);

                                                $if(main.it->valid()) {
                                                    shift_result = reconnect_shift(_pipeline, main.ray.ray->origin(), main.it->p(), shifted.it->p(), main.it->ng());
                                                } $else {
                                                    environment_connection = true;
                                                    shift_result = environment_shift(_pipeline, main.ray.ray, shifted.it->p());
                                                };

                                                auto shift_failed_flag = def(true);
                                                $if(!shift_result.success) {
                                                    shifted.alive = false;
                                                    shift_failed_flag = true;
                                                };

                                                $if(!shift_failed_flag) {
                                                    auto incoming_direction = -shifted.ray.ray->direction();
                                                    auto outgoing_direction = shift_result.wo;

                                                    // TODO strict normal check
                                                    auto shifted_bsdf_pdf = def(0.f);
                                                    _pipeline->surfaces().dispatch(shifted.it->shape()->surface_tag(), [&](auto surface) noexcept {
                                                        auto closure = surface->closure(shifted.it, swl, 1.f, time);
                                                        auto shifted_bsdf_eval = closure->evaluate(incoming_direction, outgoing_direction); // TODO check
                                                        shifted.throughput *= shifted_bsdf_eval.f * shift_result.jacobian;
                                                        shifted.pdf *= shifted_bsdf_eval.pdf * shift_result.jacobian;
                                                        shifted_bsdf_pdf = shifted_bsdf_eval.pdf;
                                                    });

                                                    shifted.connection_status = (uint) RayConnection::RAY_RECENTLY_CONNECTED;

                                                    $if(main_hit_emitter /*| has subsurface TODO*/ ) {
                                                        SampledSpectrum shifted_emitter_radiance{ swl.dimension(), 0.f };
                                                        auto shifted_lum_pdf = def(0.f);

                                                        $if(main.it->valid()) {
                                                            $if(main_hit_emitter) {
                                                                // Check if correct TODO
                                                                // From shift.p -> main.p
                                                                auto eval = _light_sampler->evaluate_hit(*main.it, shifted.it->p(), swl, time);
                                                                shifted_emitter_radiance = eval.L;
                                                                shifted_lum_pdf = eval.pdf;
                                                            };

                                                            // TODO subsurface
                                                        } $else {
                                                            shifted_emitter_radiance = main_emitter_radiance;
                                                            shifted_lum_pdf = main_lum_pdf;
                                                        };

                                                        auto shifted_weight_denominator = (shifted_previous_pdf * shifted_previous_pdf) * (
                                                            shifted_lum_pdf * shifted_lum_pdf + shifted_bsdf_pdf * shifted_bsdf_pdf
                                                        );

                                                        weight = main_weight_numerator / (D_EPSILON + shifted_weight_denominator + main_weight_denominator);

                                                        main_contribution = main.throughput * main_emitter_radiance;
                                                        shifted_contribution = shifted.throughput * shifted_emitter_radiance;
                                                    };
                                                };
                                            };
                                        } $else {
                                            // Half-vector shift
                                            auto tangent_space_incoming_direction = shifted.it->shading().world_to_local(-shifted.ray.ray->direction());
                                            auto tangent_space_outgoing_direction = def(make_float3(0.f));
                                            SampledSpectrum shifted_emitter_radiance{ swl.dimension(), 0.f };

                                            // Deny shifts between Dirac and non-Dirac BSDFs. TODO

                                            // TODO check if wo is wo
                                            auto main_bsdf_eta = def(0.f); // eta at previous main it
                                            _pipeline->surfaces().dispatch(previous_main_it.shape()->surface_tag(), [&](auto surface) noexcept {
                                                auto closure = surface->closure(make_shared<Interaction>(previous_main_it), swl, 1.f, time);
                                                main_bsdf_eta = closure->eta().value_or(1.f);
                                            });
                                            auto shifted_bsdf_eta = def(0.f); // eta at previous main it
                                            _pipeline->surfaces().dispatch(shifted.it->shape()->surface_tag(), [&](auto surface) noexcept {
                                                auto closure = surface->closure(shifted.it, swl, 1.f, time);
                                                shifted_bsdf_eta = closure->eta().value_or(1.f);
                                            });
                                            auto shift_result = half_vector_shift(
                                                main_bsdf_result.wo, main_bsdf_result.sample.wi,
                                                tangent_space_incoming_direction,
                                                main_bsdf_eta, shifted_bsdf_eta
                                            );

                                            // TODO  BSDF:EDelta

                                            auto shift_failed_flag = def(false);
                                            $if(shift_result.success) {
                                                shifted.throughput *= shift_result.jacobian;
                                                shifted.pdf *= shift_result.jacobian;
                                                tangent_space_outgoing_direction = shift_result.wo;
                                            } $else {
                                                shifted.alive = false;
                                                shift_failed_flag = true;
                                            };

                                            auto outgoing_direction = shifted.it->shading().local_to_world(tangent_space_outgoing_direction);
                                            $if(!shift_failed_flag) {
                                                _pipeline->surfaces().dispatch(shifted.it->shape()->surface_tag(), [&](auto surface) noexcept {
                                                    auto closure = surface->closure(shifted.it, swl, 1.f, time);
                                                    auto eval = closure->evaluate(tangent_space_incoming_direction, tangent_space_outgoing_direction);
                                                    
                                                    shifted.pdf *= eval.pdf;
                                                    shifted.throughput *= eval.f;
                                                });
                                                shift_failed_flag = shifted.pdf == 0.f;
                                                // Strict normal TODO
                                            };

                                            $if(!shift_failed_flag) {
                                                auto shifted_vertex_type = get_vertex_type(_pipeline, shifted.it, _config, swl, time);
                                                shifted.ray.ray = make_ray(shifted.it->p(), outgoing_direction);
                                                shifted.it = _pipeline->geometry()->intersect(shifted.ray.ray);

                                                $if(!shifted.it->valid()) {
                                                    if(!_pipeline->environment()) {
                                                        shifted.alive = false;
                                                        shift_failed_flag = true;
                                                    } else {
                                                        $if(main.it->valid()) {
                                                            shifted.alive = false;
                                                            shift_failed_flag = true;
                                                        } $elif(main_vertex_type == (uint) VertexType::VERTEX_TYPE_DIFFUSE & shifted_vertex_type == (uint) VertexType::VERTEX_TYPE_DIFFUSE) {
                                                            shifted.alive = false;
                                                            shift_failed_flag = true;
                                                        } $else {
                                                            auto eval = _light_sampler->evaluate_miss(shifted.ray.ray->direction(), swl, time);
                                                            shifted_emitter_radiance = eval.L;
                                                            postponed_shift_end = true;
                                                        };
                                                    }
                                                } $else {
                                                    $if(!main.it->valid()) {
                                                        shifted.alive = false;
                                                        shift_failed_flag = true;
                                                    } $else {
                                                        auto shifted_next_vertex_type = get_vertex_type(_pipeline, shifted.it, _config, swl, time);
                                                        $if(main_vertex_type == (uint) VertexType::VERTEX_TYPE_DIFFUSE &
                                                            shifted_vertex_type == (uint) VertexType::VERTEX_TYPE_DIFFUSE &
                                                            shifted_next_vertex_type == (uint) VertexType::VERTEX_TYPE_DIFFUSE
                                                        ) {
                                                            shifted.alive = false;
                                                            shift_failed_flag = true;
                                                        } $else {
                                                            $if(shifted.it->shape()->has_light()) {
                                                                auto eval = _light_sampler->evaluate_hit(*shifted.it, shifted.ray.ray->origin(), swl, time);
                                                                shifted_emitter_radiance = eval.L;
                                                            };
                                                            // TODO subsurface
                                                        };
                                                    };
                                                };
                                                // half vector shifted failed should go here
                                                $if(shifted.alive) {
                                                    weight = main.pdf / (shifted.pdf * shifted.pdf + main.pdf * main.pdf);
                                                    main_contribution = main.throughput * main_emitter_radiance;
                                                    shifted_contribution = shifted_contribution * shifted_emitter_radiance;
                                                } $else {
                                                    weight = 1.f / main.pdf;
                                                    main_contribution = main.throughput * main_emitter_radiance;
                                                    shifted_contribution = SampledSpectrum{ swl.dimension(), 0.f };

                                                    // Disable the failure detection below since the failure was already handled.
                                                    shifted.alive = true;
                                                    postponed_shift_end = true;
                                                };
                                            };
                                        };
                                    };
                                };
                            };

                            // shift failed should go here
                            $if(!shifted.alive) {
                                weight = main_weight_numerator / (D_EPSILON + main_weight_denominator);
                                main_contribution = main.throughput * main_emitter_radiance;
                                shifted_contribution = SampledSpectrum{ swl.dimension(), 0.f };
                            };

                            $if(depth + 1 >= _config->m_max_depth) {
                                main.add_radiance(main_contribution, weight);
                                shifted.add_radiance(shifted_contribution, weight);
                                shifted.add_gradient(shifted_contribution - main_contribution, weight);
                            };

                            shifted.alive = ite(postponed_shift_end, false, shifted.alive);
                        }

			            // Stop if the base path hit the environment.
                        // TODO main.rRec.type
                        $if(!main.it->valid() /*| !(main.rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance)*/) {
                            $break;
                        };

                        depth += 1;
                        $if(depth > _config->m_rr_depth) {
                            // Russian Roulette
                            auto q = min((main.throughput / main.pdf).max() * main.eta * main.eta, 0.95f);
                            $if(_sampler->generate_1d() >= q) {
                                $break;
                            };

                            main.pdf *= q;
                            for(int i = 0; i < 4; i++) {
                                shifteds[i].pdf *= q;
                            }
                        };
                    };
                };
            };
        };

        return result; 
    };
};

void GradientPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Camera::Instance *camera) noexcept {
    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    auto pixel_count = resolution.x * resolution.y;
    sampler()->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << compute::synchronize();

    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    using namespace luisa::compute;

    GPTConfig config{
        .m_max_depth = node<GradientPathTracing>()->max_depth(),
        .m_rr_depth = node<GradientPathTracing>()->rr_depth(),
        .m_shift_threshold = 0.1f // TODO
    };

    GradientPathTracingImpl impl(
        &pipeline(),
        sampler(),
        light_sampler(),
        camera,
        &config
    );
    float diff_scale_factor = 1.0f / std::sqrt(spp);

    Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel_id = dispatch_id().xy();
        auto eval = impl.evaluate_point(pixel_id, frame_index, time, diff_scale_factor);
        // auto L = Li(camera, frame_index, pixel_id, time);
        camera->film()->accumulate(pixel_id, shutter_weight * pipeline().spectrum()->srgb(eval.swl, eval.very_direct));
    };

    Clock clock_compile;
    auto render = pipeline().device().compile(render_kernel);
    auto integrator_shader_compilation_time = clock_compile.toc();
    LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
    auto shutter_samples = camera->node()->shutter_samples();
    command_buffer << synchronize();

    LUISA_INFO("Rendering started.");
    Clock clock;
    ProgressBar progress;
    progress.update(0.);
    auto dispatch_count = 0u;
    auto sample_id = 0u;
    for (auto s : shutter_samples) {
        pipeline().update(command_buffer, s.point.time);
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            auto dispatches_per_commit = 32u;
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                dispatch_count = 0u;
                auto p = sample_id / static_cast<double>(spp);
                command_buffer << [&progress, p] { progress.update(p); };
            }
        }
    }
    command_buffer << synchronize();
    progress.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);    
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GradientPathTracing)
