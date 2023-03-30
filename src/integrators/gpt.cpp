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
#include <base/display.h>

namespace luisa::render {

using namespace compute;

class GradientPathTracing final : public ProgressiveIntegrator {
private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    float _shift_threshold;
    bool _central_radiance;

public:
    GradientPathTracing(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _shift_threshold{std::max(desc->property_float_or_default("shift_threshold", 0.1f), 0.0f)},
          _central_radiance{desc->property_bool_or_default("central_radiance", false)} {}
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] auto shift_threshold() const noexcept { return _shift_threshold; }
    [[nodiscard]] auto central_radiance() const noexcept { return _central_radiance; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class GradientPathTracingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;

private:
    struct Evaluation {
        SampledSpectrum very_direct;
        SampledSpectrum throughput;
        SampledSpectrum gradients[4];
        SampledSpectrum neighbor_throughput[4];
        SampledWavelengths swl;
    };

    const float D_EPSILON = 1e-8f;

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
        SampledSpectrum weight;// throughput / pdf
        Float pdf_div_main_pdf;// shifted.pdf / main.pdf
        SampledSpectrum radiance;
        SampledSpectrum gradient;
        // RadianceQueryRecord rRec;        ///< The radiance query record for this ray.
        luisa::shared_ptr<Interaction> it;
        Float eta;// Current refractive index
        Bool alive;
        UInt connection_status;

        explicit RayState(uint dimension) : radiance(dimension, 0.0f), gradient(dimension, 0.0f), it(luisa::make_shared<Interaction>()), eta(1.0f), pdf(1.0f),
                                            weight(dimension, 0.f), pdf_div_main_pdf(1.f), throughput(dimension, 0.0f), alive(true), connection_status((uint)RAY_NOT_CONNECTED) {}

        inline void add_radiance(const SampledSpectrum &contribution) noexcept {
            radiance += contribution;
        }

        inline void add_gradient(const SampledSpectrum &contribution) noexcept {
            gradient += contribution;
        }
    };

    const float epsilon = 1e-4f;
    const float shadow_epsilon = 1e-3f;

    struct HalfVectorShiftResult {
        Bool success;
        Float jacobian;
        Float3 wo;
    };

    struct ReconnectionShiftResult {
        Bool success;
        Float jacobian;
        Float3 wo;
    };

    struct SurfaceSampleResult {
        Surface::Sample sample;
        SampledSpectrum weight;
        Float pdf;
        Float3 wo;
        Float eta;
    };

    const uint2 pixel_shifts[4] = {
        make_uint2(1, 0), // right
        make_uint2(0, 1), // bottom
        make_uint2(-1, 0),// left
        make_uint2(0, -1) // top
    };

    [[nodiscard]] auto test_visibility(Expr<float3> point1, Expr<float3> point2) const noexcept;
    [[nodiscard]] auto test_environment_visibility(const Var<Ray> &ray) const noexcept;
    [[nodiscard]] auto get_vertex_type_by_roughness(Expr<float> roughness) const noexcept;
    [[nodiscard]] auto get_vertex_type(luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl, Expr<float> time) const noexcept;
    [[nodiscard]] auto half_vector_shift(
        Float3 tangent_space_main_wi,
        Float3 tangent_space_main_wo,
        Float3 tangent_space_shifted_wi,
        Float main_eta, Float shifted_eta) const noexcept;
    [[nodiscard]] auto reconnect_shift(
        Expr<float3> main_source_vertex,
        Expr<float3> target_vertex,
        Expr<float3> shift_source_vertex,
        Expr<float3> target_normal) const noexcept;
    [[nodiscard]] auto environment_shift(
        const Var<Ray> &main_ray,
        Expr<float3> shift_source_vertex) const noexcept;

    [[nodiscard]] auto evaluate_point(Expr<uint2> pixel_coord, Expr<uint> sample_index, Expr<float> time, float diff_scale_factor, const Camera::Instance *_camera) const noexcept;
    [[nodiscard]] auto sample_surface(RayState &state, SampledWavelengths &swl, Expr<float> time) const noexcept;
    [[nodiscard]] SampledSpectrum evaluate(RayState &main, RayState *shifteds, SampledWavelengths &swl, Expr<float> time, Expr<uint2> pixel_id) const noexcept;

protected:
    void _render_one_camera(CommandBuffer &command_buffer,
                            Camera::Instance *camera) noexcept override;

    [[nodiscard]] Float3 Li(const Camera::Instance *camera,
                            Expr<uint> frame_index,
                            Expr<uint2> pixel_id,
                            Expr<float> time) const noexcept override;
};

class ImageBuffer {

private:
    Pipeline &_pipeline;
    Buffer<float> _image;
    uint2 _resolution;

private:
    static constexpr auto clear_shader_name = luisa::string_view{"__gpt_image_buffer_clear_shader"};

public:
    ImageBuffer(Pipeline &pipeline, uint2 resolution, bool enabled = true) noexcept
        : _pipeline{pipeline}, _resolution{resolution} {
        _pipeline.register_shader<1u>(
            clear_shader_name, [resolution](BufferFloat image) noexcept {
                image.write(dispatch_x(), 0.f);
            });
        if (enabled) {
            _image = pipeline.device().create_buffer<float>(
                resolution.x * resolution.y * 4u);
        }
    }
    void clear(CommandBuffer &command_buffer) const noexcept {
        if (_image) {
            command_buffer << _pipeline.shader<1u, Buffer<float>>(clear_shader_name, _image)
                                  .dispatch(_image.size());
        }
    }
    [[nodiscard]] auto save(CommandBuffer &command_buffer,
                            std::filesystem::path path) const noexcept
        -> luisa::function<void()> {
        if (!_image) { return {}; }
        auto host_image = luisa::make_shared<luisa::vector<float4>>();
        host_image->resize(_resolution.x * _resolution.y);
        command_buffer << _image.copy_to(host_image->data());
        return [host_image, size = _resolution, path = std::move(path)] {
            for (auto &p : *host_image) {
                p = make_float4(p.xyz() / p.w, 1.f);
            }
            LUISA_INFO("Saving auxiliary buffer to '{}'.", path.string());
            save_image(path.string(), reinterpret_cast<const float *>(host_image->data()), size, 4);
        };
    }
    void accumulate(Expr<uint2> p, Expr<float3> value, Expr<float> effective_spp = 1.f) noexcept {
        if (_image) {
            $if(!any(isnan(value))) {
                auto index = p.y * _resolution.x + p.x;
                auto v = make_float4(value, effective_spp);
                for (auto ch = 0u; ch < 4u; ch++) {
                    _image.atomic(index * 4u + ch).fetch_add(v[ch]);
                }
            };
        }
    }
};

luisa::unique_ptr<Integrator::Instance> GradientPathTracing::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<GradientPathTracingInstance>(
        pipeline, command_buffer, this);
}

[[nodiscard]] auto GradientPathTracingInstance::test_visibility(Expr<float3> point1, Expr<float3> point2) const noexcept {
    auto shadow_ray = make_ray(
        point1, normalize(point2 - point1), epsilon, (1.f - shadow_epsilon) * length(point2 - point1));
    return !pipeline().geometry()->intersect_any(shadow_ray);
}

[[nodiscard]] auto GradientPathTracingInstance::test_environment_visibility(const Var<Ray> &ray) const noexcept {
    if (!pipeline().environment()) return def(false);
    auto shadow_ray = make_ray(
        ray->origin(), ray->direction(), epsilon, std::numeric_limits<float>::max());
    return !pipeline().geometry()->intersect_any(shadow_ray);
}

[[nodiscard]] auto GradientPathTracingInstance::get_vertex_type_by_roughness(Expr<float> roughness) const noexcept {
    return ite(roughness <= node<GradientPathTracing>()->shift_threshold(), (uint)VertexType::VERTEX_TYPE_GLOSSY, (uint)VertexType::VERTEX_TYPE_DIFFUSE);
}

[[nodiscard]] auto GradientPathTracingInstance::get_vertex_type(luisa::shared_ptr<Interaction> it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto surface_tag = it->shape().surface_tag();
    Float2 roughness;
    pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
        auto closure = surface->closure(it, swl, make_float3(0.f, 0.f, 1.f), 1.f, time);// TODO fix
        roughness = closure->roughness();
    });
    return get_vertex_type_by_roughness(min(roughness.x, roughness.y));
}

[[nodiscard]] auto GradientPathTracingInstance::half_vector_shift(
    Float3 tangent_space_main_wi,
    Float3 tangent_space_main_wo,
    Float3 tangent_space_shifted_wi,
    Float main_eta, Float shifted_eta) const noexcept {
    HalfVectorShiftResult result;

    $if(cos_theta(tangent_space_main_wi) * cos_theta(tangent_space_shifted_wi) < 0.f) {
        // Refraction

        $if(main_eta == 1.f | shifted_eta == 1.f) {
            result.success = false;
        }
        $else {
            auto tangent_space_half_vector_non_normalized_main = ite(
                cos_theta(tangent_space_main_wi) < 0.f,
                -(tangent_space_main_wi * main_eta + tangent_space_main_wo),
                -(tangent_space_main_wi + tangent_space_main_wo * main_eta));

            auto tangent_space_half_vector = normalize(tangent_space_half_vector_non_normalized_main);

            Float3 tangent_space_shifted_wo;
            auto refract_not_internal = refract(tangent_space_shifted_wi,
                                                tangent_space_half_vector,
                                                shifted_eta,
                                                std::addressof(tangent_space_shifted_wo));

            $if(!refract_not_internal) {
                result.success = false;
            }
            $else {
                auto tangent_space_half_vector_non_normalized_shifted = ite(
                    cos_theta(tangent_space_shifted_wi) < 0.f,
                    -(tangent_space_shifted_wi * shifted_eta + tangent_space_shifted_wo),
                    -(tangent_space_shifted_wi + tangent_space_shifted_wo * shifted_eta));

                auto h_length_squared = length_squared(tangent_space_half_vector_non_normalized_shifted) / (D_EPSILON + length_squared(tangent_space_half_vector_non_normalized_main));
                auto wo_dot_h = abs(dot(tangent_space_main_wo, tangent_space_half_vector)) / (D_EPSILON + abs(dot(tangent_space_shifted_wo, tangent_space_half_vector)));

                result.success = true;
                result.wo = tangent_space_shifted_wo;
                result.jacobian = h_length_squared * wo_dot_h;
            };

            result.success = ite(cos_theta(tangent_space_shifted_wi) * cos_theta(tangent_space_shifted_wo) >= 0.f, false, result.success);// TODO check reject
        };
    }
    $else {
        // Reflection
        auto tangent_space_half_vector = normalize(tangent_space_main_wi + tangent_space_main_wo);
        auto tangent_space_shifted_wo = reflect(tangent_space_shifted_wi, tangent_space_half_vector);

        auto wo_dot_h = abs(dot(tangent_space_shifted_wo, tangent_space_half_vector)) / abs(dot(tangent_space_main_wo, tangent_space_half_vector));

        result.success = true;
        result.wo = tangent_space_shifted_wo;
        result.jacobian = wo_dot_h;

        result.success = ite(cos_theta(tangent_space_shifted_wi) * cos_theta(tangent_space_shifted_wo) <= 0.f, false, result.success);// TODO check reject
    };

    return result;
}

[[nodiscard]] auto GradientPathTracingInstance::reconnect_shift(
    Expr<float3> main_source_vertex,
    Expr<float3> target_vertex,
    Expr<float3> shift_source_vertex,
    Expr<float3> target_normal) const noexcept {
    ReconnectionShiftResult result;
    result.success = false;
    $if(test_visibility(shift_source_vertex, target_vertex)) {
        auto main_edge = main_source_vertex - target_vertex;
        auto shifted_edge = shift_source_vertex - target_vertex;

        auto main_edge_length_squared = length_squared(main_edge);
        auto shifted_edge_length_squared = length_squared(shifted_edge);

        auto shifted_wo = -shifted_edge / sqrt(shifted_edge_length_squared);

        auto main_opposing_cosine = dot(main_edge, target_normal) / sqrt(main_edge_length_squared);
        auto shifted_opposing_cosine = dot(shifted_wo, target_normal);

        auto jacobian = abs(shifted_opposing_cosine * main_edge_length_squared) / abs(main_opposing_cosine * shifted_edge_length_squared);

        result.success = true;
        result.jacobian = jacobian;
        result.wo = shifted_wo;
    };

    return result;
}

[[nodiscard]] auto GradientPathTracingInstance::environment_shift(
    const Var<Ray> &main_ray,
    Expr<float3> shift_source_vertex) const noexcept {
    ReconnectionShiftResult result;
    result.success = false;

    auto offset_ray = make_ray(
        shift_source_vertex, main_ray->direction(), main_ray->t_min(), main_ray->t_max());

    $if(test_environment_visibility(offset_ray)) {
        result.success = true;
        result.jacobian = 1.f;
        result.wo = main_ray->direction();
    };

    return result;
}

// Entrance function of GPT
[[nodiscard]] auto GradientPathTracingInstance::evaluate_point(Expr<uint2> pixel_coord, Expr<uint> sample_index, Expr<float> time, float diff_scale_factor, const Camera::Instance *_camera) const noexcept {
    sampler()->start(pixel_coord, sample_index);
    auto u_filter = sampler()->generate_pixel_2d();
    auto u_lens = _camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
    auto [main_ray_diff, _, main_ray_weight] = _camera->generate_ray_differential(pixel_coord, time, u_filter, u_lens);
    auto spectrum = pipeline().spectrum();
    auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());

    RayState main_ray{swl.dimension()};
    main_ray.ray = main_ray_diff;
    main_ray.ray.scale_differential(diff_scale_factor);
    main_ray.throughput = SampledSpectrum{swl.dimension(), main_ray_weight};
    main_ray.weight = SampledSpectrum{swl.dimension(), main_ray_weight};

    RayState shifted_rays[4] = {
        RayState{swl.dimension()},
        RayState{swl.dimension()},
        RayState{swl.dimension()},
        RayState{swl.dimension()}};

    for (int i = 0; i < 4; i++) {
        auto [shifted_diff, _, shifted_weight] = _camera->generate_ray_differential(pixel_coord + pixel_shifts[i], time, u_filter, u_lens);
        shifted_rays[i].ray = shifted_diff;
        shifted_rays[i].ray.scale_differential(diff_scale_factor);
        shifted_rays[i].throughput = SampledSpectrum{swl.dimension(), shifted_weight};
        shifted_rays[i].weight = SampledSpectrum{swl.dimension(), shifted_weight};
    }

    // Actual implementation
    SampledSpectrum very_direct = evaluate(main_ray, shifted_rays, swl, time, pixel_coord);

    return Evaluation{
        .very_direct = very_direct,
        .throughput = main_ray.radiance,
        .gradients = {
            shifted_rays[0].gradient,
            shifted_rays[1].gradient,
            shifted_rays[2].gradient,
            shifted_rays[3].gradient},
        .neighbor_throughput = {shifted_rays[0].radiance, shifted_rays[1].radiance, shifted_rays[2].radiance, shifted_rays[3].radiance},
        .swl = swl};
}

[[nodiscard]] auto GradientPathTracingInstance::sample_surface(RayState &state, SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto &it = state.it;
    auto &ray = state.ray;

    SurfaceSampleResult result{
        .sample = Surface::Sample::zero(swl.dimension()),
        .weight = SampledSpectrum{swl.dimension(), 0.f},
        .pdf = def(0.f),
        .eta = def(0.f)};
    auto surface_tag = it->shape().surface_tag();
    auto u_lobe = sampler()->generate_1d();
    auto u_bsdf = sampler()->generate_2d();
    pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
        auto closure = surface->closure(it, swl, -ray.ray->direction(), 1.f, time);
        result.sample = closure->sample(-ray.ray->direction(), u_lobe, u_bsdf);
        result.eta = closure->eta().value_or(1.f);
    });
    result.weight = result.sample.eval.f;
    result.pdf = result.sample.eval.pdf;
    result.wo = -ray.ray->direction();

    return result;
}

[[nodiscard]] SampledSpectrum GradientPathTracingInstance::evaluate(RayState &main, RayState *shifteds, SampledWavelengths &swl, Expr<float> time, Expr<uint2> pixel_id) const noexcept {
    SampledSpectrum result{swl.dimension(), 0.f};

    *main.it = *pipeline().geometry()->intersect(main.ray.ray);
    // main.ray.ray->set_t_min(epsilon);

    for (int i = 0; i < 4; i++) {
        auto &shifted = shifteds[i];
        *shifted.it = *pipeline().geometry()->intersect(shifted.ray.ray);
        // shifted.ray.ray->set_t_min(epsilon);
    }

    $if(!main.it->valid()) {
        if (pipeline().environment()) {
            auto eval = light_sampler()->evaluate_miss(main.ray.ray->direction(), swl, time);
            // result += main.throughput * eval.L;
            result += main.weight * eval.L;
        };
    }
    $else {
        if (!pipeline().lights().empty()) {
            $if(main.it->shape().has_light()) {
                auto eval = light_sampler()->evaluate_hit(*main.it, main.ray.ray->origin(), swl, time);
                // result += main.throughput * eval.L;
                result += main.weight * eval.L;
            };
        }
        // Subsurface omitted TODO

        for (int i = 0; i < 4; i++) {
            auto &shifted = shifteds[i];
            $if(!shifted.it->valid()) {
                shifted.alive = false;
            };
        }

        // Strict normal check to produce the same results as bidirectional methods when normal mapping is used.
        // TODO

        // Main PT Loop
        $for(depth, node<GradientPathTracing>()->max_depth()) {
            // Strict normal check to produce the same results as bidirectional methods when normal mapping is used.
            // TODO

            auto last_segment = depth + 1 == node<GradientPathTracing>()->max_depth();
            $if(!main.it->shape().has_surface()) { $break; };

            //
            // Direct Illumination Sampling
            //

            // Sample incoming radiance from lights (next event estimation)
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto main_light_sample = light_sampler()->sample(*main.it, u_light_selection, u_light_surface, swl, time);
            auto main_occluded = pipeline().geometry()->intersect_any(main_light_sample.shadow_ray);

            auto main_surface_tag = main.it->shape().surface_tag();
            auto wo = -main.ray.ray->direction();
            $if(main_light_sample.eval.pdf > 0.f & !main_occluded) {
                auto wi = main_light_sample.shadow_ray->direction();

                Surface::Evaluation main_light_eval{.f = SampledSpectrum{swl.dimension(), 0.f}, .pdf = 0.f};
                pipeline().surfaces().dispatch(main_surface_tag, [&](auto surface) noexcept {
                    auto closure = surface->closure(main.it, swl, wo, 1.f, time);
                    main_light_eval = closure->evaluate(wo, wi);
                });

                auto main_distance_squared = length_squared(main.it->p() - main_light_sample.eval.p);
                auto main_opposing_cosine = dot(main_light_sample.eval.ng, main.it->p() - main_light_sample.eval.p) / sqrt(main_distance_squared);

                auto main_bsdf_pdf = main_light_eval.pdf;

                // Balance Heuristic
                auto main_weight = main.weight / (main_light_sample.eval.pdf + main_bsdf_pdf);// main.throughput / (main.pdf * (light sample pdf + bsdf pdf))

                if (node<GradientPathTracing>()->central_radiance()) {
                    main.add_radiance(main_weight * main_light_eval.f * main_light_sample.eval.L);
                }

                // strict normal not implemented TODO
                for (int i = 0; i < 4; i++) {
                    auto &shifted = shifteds[i];
                    SampledSpectrum main_contribution{swl.dimension(), 0.f};
                    SampledSpectrum shifted_contribution{swl.dimension(), 0.f};

                    auto shift_successful = shifted.alive;

                    $if(shift_successful) {
                        $switch(shifted.connection_status) {
                            $case((uint)RayConnection::RAY_CONNECTED) {
                                auto shifted_bsdf_pdf = main_bsdf_pdf;
                                auto shifted_emitter_pdf = main_light_sample.eval.pdf;
                                auto shifted_bsdf_value = main_light_eval.f;
                                auto shifted_emitter_radiance = main_light_sample.eval.L;
                                auto jacobian = 1.f;

                                // MIS between main and shifted
                                auto new_denominator = main_light_sample.eval.pdf + main_bsdf_pdf + jacobian * shifted.pdf_div_main_pdf * (shifted_bsdf_pdf + shifted_emitter_pdf);
                                main_contribution = main_light_eval.f * main_light_sample.eval.L * main.weight / new_denominator;
                                shifted_contribution = jacobian * (shifted_bsdf_value * shifted_emitter_radiance) * shifted.weight * shifted.pdf_div_main_pdf / new_denominator;
                            };
                            $case((uint)RayConnection::RAY_RECENTLY_CONNECTED) {
                                auto incoming_direction = normalize(shifted.it->p() - main.it->p());

                                Surface::Evaluation shifted_bsdf_eval{.f = SampledSpectrum{swl.dimension(), 0.f}, .pdf = 0.f};
                                pipeline().surfaces().dispatch(main_surface_tag, [&](auto surface) noexcept {
                                    // TODO check if incoming correct
                                    auto closure = surface->closure(main.it, swl, incoming_direction, 1.f, time);
                                    shifted_bsdf_eval = closure->evaluate(incoming_direction, main_light_sample.shadow_ray->direction());
                                });
                                auto shifted_emitter_pdf = main_light_sample.eval.pdf;
                                auto shifted_bsdf_value = shifted_bsdf_eval.f;
                                auto shifted_bsdf_pdf = ite(!main_occluded, shifted_bsdf_eval.pdf, 0.f);
                                auto shifted_emitter_radiance = main_light_sample.eval.L;
                                auto jacobian = 1.f;

                                // MIS between main and shifted
                                auto new_denominator = main_light_sample.eval.pdf + main_bsdf_pdf + jacobian * shifted.pdf_div_main_pdf * (shifted_bsdf_pdf + shifted_emitter_pdf);
                                main_contribution = main_light_eval.f * main_light_sample.eval.L * main.weight / new_denominator;
                                shifted_contribution = jacobian * (shifted_bsdf_value * shifted_emitter_radiance) * shifted.weight * shifted.pdf_div_main_pdf / new_denominator;
                            };
                            $case((uint)RayConnection::RAY_NOT_CONNECTED) {
                                // TODO
                                auto main_vertex_type = get_vertex_type(main.it, swl, time);
                                auto shifted_vertex_type = get_vertex_type(shifted.it, swl, time);

                                $if(main_vertex_type == (uint)VertexType::VERTEX_TYPE_DIFFUSE & shifted_vertex_type == (uint)VertexType::VERTEX_TYPE_DIFFUSE) {
                                    auto shifted_light_sample = light_sampler()->sample(*shifted.it, u_light_selection, u_light_surface, swl, time);
                                    auto shifted_occluded = pipeline().geometry()->intersect_any(shifted_light_sample.shadow_ray);

                                    auto shifted_emitter_radiance = shifted_light_sample.eval.L;
                                    auto shifted_emitter_pdf = shifted_light_sample.eval.pdf;

                                    auto shifted_distance_squared = length_squared(shifted.it->p() - shifted_light_sample.eval.p);
                                    auto emitter_direction = (shifted.it->p() - shifted_light_sample.eval.p) / sqrt(shifted_distance_squared);
                                    auto shifted_opposing_cosine = -dot(shifted_light_sample.eval.ng, emitter_direction);

                                    // TODO: No strict normal here
                                    auto shifted_surface_tag = shifted.it->shape().surface_tag();
                                    Surface::Evaluation shifted_light_eval{.f = SampledSpectrum{swl.dimension(), 0.f}, .pdf = 0.f};
                                    pipeline().surfaces().dispatch(shifted_surface_tag, [&](auto surface) noexcept {
                                        auto closure = surface->closure(shifted.it, swl, -shifted.ray.ray->direction(), 1.f, time);
                                        shifted_light_eval = closure->evaluate(-shifted.ray.ray->direction(), -emitter_direction);
                                    });

                                    auto shifted_bsdf_value = shifted_light_eval.f;
                                    auto shifted_bsdf_pdf = ite(!shifted_occluded, shifted_light_eval.pdf, 0.f);
                                    auto jacobian = abs(shifted_opposing_cosine * main_distance_squared) / (D_EPSILON + abs(main_opposing_cosine * shifted_distance_squared));

                                    // MIS between main and shifted
                                    auto new_denominator = main_light_sample.eval.pdf + main_bsdf_pdf + jacobian * shifted.pdf_div_main_pdf * (shifted_bsdf_pdf + shifted_emitter_pdf);
                                    main_contribution = main_light_eval.f * main_light_sample.eval.L * main.weight / new_denominator;
                                    shifted_contribution = jacobian * (shifted_bsdf_value * shifted_emitter_radiance) * shifted.weight * shifted.pdf_div_main_pdf / new_denominator;
                                };
                            };
                        };
                    }
                    $else {// shift_successful == false
                        main_contribution = main_weight * main_light_eval.f * main_light_sample.eval.L;
                        shifted_contribution = SampledSpectrum{swl.dimension(), 0.f};
                    };

                    if (!node<GradientPathTracing>()->central_radiance()) {
                        main.add_radiance(main_contribution);
                        shifted.add_radiance(shifted_contribution);
                        shifted.add_gradient(shifted_contribution - main_contribution);
                    }
                }
            };

            //
            // BSDF Sampling & Emitter
            //

            auto main_bsdf_result = sample_surface(main, swl, time);
            $if(main_bsdf_result.pdf <= 0.f) {
                $break;
            };
            auto main_wo = main_bsdf_result.sample.wi;

            auto main_wo_dot_ng = dot(main.it->ng(), main_wo);
            // TODO: strict normal

            auto previous_main_it = *main.it;
            auto previous_main_ray = main.ray;

            auto main_hit_emitter = def(false);
            auto main_emitter_radiance = SampledSpectrum{swl.dimension(), 0.f};
            auto main_emitter_pdf = def(0.f);

            auto main_vertex_type = get_vertex_type(main.it, swl, time);
            auto main_next_vertex_type = def(0u);

            main.ray = RayDifferential{.ray = main.it->spawn_ray(main_wo)};
            *main.it = *pipeline().geometry()->intersect(main.ray.ray);
            $if(main.it->valid()) {
                if (!pipeline().lights().empty()) {
                    $if(main.it->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*main.it, main.ray.ray->origin(), swl, time);
                        main_emitter_radiance = eval.L;
                        main_emitter_pdf = eval.pdf;
                        main_hit_emitter = true;
                    };
                }

                // TODO: subsurface scattering

                main_next_vertex_type = get_vertex_type(main.it, swl, time);
            }
            $else {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(main.ray.ray->direction(), swl, time);
                    main_emitter_radiance = eval.L;
                    main_emitter_pdf = eval.pdf;
                    main_hit_emitter = true;

                    main_next_vertex_type = (uint)VERTEX_TYPE_DIFFUSE;
                } else {// hit nothing
                    $break;
                }
            };

            // Continue the shift
            auto main_bsdf_pdf = main_bsdf_result.pdf;
            auto main_previous_weight = main.weight;

            main.weight *= main_bsdf_result.sample.eval.f / main_bsdf_result.pdf;
            main.eta *= main_bsdf_result.eta;

            // auto main_lum_pdf = ite(
            //     main_hit_emitter & depth + 1u >= node<GradientPathTracing>()->min_depth /* TODO :& !(mainBsdfResult.bRec.sampledType & BSDF::EDelta)*/,
            //     main_emitter_pdf, 0.f);
            auto main_lum_pdf = main_emitter_pdf;

            auto main_weight = main_previous_weight / (main_lum_pdf + main_bsdf_pdf);// TODO_EPSILON

            if (node<GradientPathTracing>()->central_radiance()) {
                main.add_radiance(main_emitter_radiance * main_weight * main_bsdf_result.sample.eval.f);
            }

            for (int i = 0; i < 4; i++) {
                auto &shifted = shifteds[i];

                SampledSpectrum shifted_emitter_radiance{swl.dimension(), 0.f};
                SampledSpectrum main_contribution{swl.dimension(), 0.f};
                SampledSpectrum shifted_contribution{swl.dimension(), 0.f};
                auto weight = def(0.f);

                auto postponed_shift_end = def(false);// Kills the shift after evaluating the current radiance.

                $if(shifted.alive) {
                    auto shifted_previous_weight = shifted.weight;
                    auto previous_shifted_pdf_div_main_pdf = shifted.pdf_div_main_pdf;
                    $switch(shifted.connection_status) {
                        $case((uint)RayConnection::RAY_CONNECTED) {
                            auto shifted_bsdf_value = main_bsdf_result.weight;
                            auto shifted_bsdf_pdf = main_bsdf_pdf;
                            auto shifted_lum_pdf = main_lum_pdf;
                            auto &shifted_emitter_radiance = main_emitter_radiance;
                            $if(shifted_bsdf_pdf <= 0.f) {
                                shifted.alive = false;
                            }
                            $else {
                                shifted.weight *= shifted_bsdf_value / shifted_bsdf_pdf;
                                shifted.pdf_div_main_pdf *= shifted_bsdf_pdf / main_bsdf_pdf;

                                // MIS between main and shifted
                                auto new_denominator = main_lum_pdf + main_bsdf_pdf + previous_shifted_pdf_div_main_pdf * (shifted_bsdf_pdf + shifted_lum_pdf);
                                main_contribution = main_bsdf_result.weight * main_light_sample.eval.L * main_previous_weight / new_denominator;
                                shifted_contribution = (shifted_bsdf_value * shifted_emitter_radiance) * shifted_previous_weight * previous_shifted_pdf_div_main_pdf / new_denominator;
                            };
                        };
                        $case((uint)RayConnection::RAY_RECENTLY_CONNECTED) {
                            auto incoming_direction = normalize(shifted.it->p() - main.ray.ray->origin());
                            Surface::Evaluation shifted_bsdf_eval{.f = SampledSpectrum{swl.dimension(), 0.f}, .pdf = 0.f};
                            pipeline().surfaces().dispatch(previous_main_it.shape().surface_tag(), [&](auto surface) noexcept {
                                auto closure = surface->closure(make_shared<Interaction>(previous_main_it), swl, incoming_direction, 1.f, time);
                                shifted_bsdf_eval = closure->evaluate(incoming_direction, main.ray.ray->direction());// TODO check if main.ray.ray right
                            });

                            auto shifted_bsdf_value = shifted_bsdf_eval.f;
                            auto shifted_bsdf_pdf = shifted_bsdf_eval.pdf;
                            auto shifted_lum_pdf = main_lum_pdf;
                            auto &shifted_emitter_radiance = main_emitter_radiance;

                            $if(shifted_bsdf_pdf <= 0.f) {
                                shifted.alive = false;
                            }
                            $else {
                                shifted.weight *= shifted_bsdf_value / shifted_bsdf_pdf;
                                shifted.pdf_div_main_pdf *= shifted_bsdf_pdf / main_bsdf_pdf;

                                shifted.connection_status = (uint)RayConnection::RAY_CONNECTED;

                                // MIS between main and shifted
                                auto new_denominator = main_lum_pdf + main_bsdf_pdf + previous_shifted_pdf_div_main_pdf * (shifted_bsdf_pdf + shifted_lum_pdf);
                                main_contribution = main_bsdf_result.weight * main_light_sample.eval.L * main_previous_weight / new_denominator;
                                shifted_contribution = (shifted_bsdf_value * shifted_emitter_radiance) * shifted_previous_weight * previous_shifted_pdf_div_main_pdf / new_denominator;
                            };
                        };
                        $case((uint)RayConnection::RAY_NOT_CONNECTED) {
                            auto shifted_vertex_type = get_vertex_type(shifted.it, swl, time);
                            $if(main_vertex_type == (uint)VertexType::VERTEX_TYPE_DIFFUSE & main_next_vertex_type == (uint)VertexType::VERTEX_TYPE_DIFFUSE & shifted_vertex_type == (uint)VertexType::VERTEX_TYPE_DIFFUSE) {
                                // Reconnect shift
                                shifted.alive = false;
                                $if(!last_segment | main_hit_emitter /*| main.it has subsurface TODO*/) {
                                    ReconnectionShiftResult shift_result;
                                    auto environment_connection = def(false);

                                    $if(main.it->valid()) {
                                        shift_result = reconnect_shift(main.ray.ray->origin(), main.it->p(), shifted.it->p(), main.it->ng());
                                    }
                                    $else {
                                        environment_connection = true;
                                        shift_result = environment_shift(main.ray.ray, shifted.it->p());
                                    };

                                    auto shift_failed_flag = def(false);
                                    $if(!shift_result.success) {
                                        shifted.alive = false;
                                        shift_failed_flag = true;
                                    };

                                    $if(!shift_failed_flag) {
                                        auto incoming_direction = -shifted.ray.ray->direction();
                                        auto outgoing_direction = shift_result.wo;

                                        // TODO strict normal check
                                        auto shifted_bsdf_pdf = def(0.f);
                                        auto shifted_bsdf_value = SampledSpectrum{swl.dimension(), 0.f};
                                        pipeline().surfaces().dispatch(shifted.it->shape().surface_tag(), [&](auto surface) noexcept {
                                            auto closure = surface->closure(shifted.it, swl, incoming_direction, 1.f, time);
                                            auto shifted_bsdf_eval = closure->evaluate(incoming_direction, outgoing_direction);// TODO check

                                            shifted_bsdf_pdf = shifted_bsdf_eval.pdf;
                                            shifted_bsdf_value = shifted_bsdf_eval.f;
                                        });

                                        $if(shifted_bsdf_pdf <= 0.f) {
                                            shifted.alive = false;
                                        }
                                        $else {

                                            shifted.weight *= shifted_bsdf_value / shifted_bsdf_pdf;
                                            shifted.pdf_div_main_pdf *= shift_result.jacobian * shifted_bsdf_pdf / main_bsdf_pdf;

                                            shifted.connection_status = (uint)RayConnection::RAY_RECENTLY_CONNECTED;

                                            $if(main_hit_emitter /*| has subsurface TODO*/) {
                                                SampledSpectrum shifted_emitter_radiance{swl.dimension(), 0.f};
                                                auto shifted_lum_pdf = def(0.f);

                                                $if(main.it->valid()) {
                                                    $if(main_hit_emitter) {
                                                        // Check if correct TODO
                                                        // From shift.p -> main.p
                                                        auto eval = light_sampler()->evaluate_hit(*main.it, shifted.it->p(), swl, time);
                                                        shifted_emitter_radiance = eval.L;
                                                        shifted_lum_pdf = eval.pdf;
                                                    };

                                                    // TODO subsurface
                                                }
                                                $else {
                                                    shifted_emitter_radiance = main_emitter_radiance;
                                                    shifted_lum_pdf = main_lum_pdf;
                                                };

                                                // MIS between main and shifted
                                                auto new_denominator = main_lum_pdf + main_bsdf_pdf + shift_result.jacobian * previous_shifted_pdf_div_main_pdf * (shifted_bsdf_pdf + shifted_lum_pdf);
                                                main_contribution = main_bsdf_result.weight * main_light_sample.eval.L * main_previous_weight / new_denominator;
                                                shifted_contribution = (shifted_bsdf_value * shifted_emitter_radiance) * shifted_previous_weight * shift_result.jacobian * previous_shifted_pdf_div_main_pdf / new_denominator;
                                            };
                                        };
                                    };
                                };
                            }
                            $else {
                                // Half-vector shift
                                auto tangent_space_incoming_direction = shifted.it->shading().world_to_local(-shifted.ray.ray->direction());
                                auto tangent_space_outgoing_direction = def(make_float3(0.f));
                                SampledSpectrum shifted_emitter_radiance{swl.dimension(), 0.f};

                                // Deny shifts between Dirac and non-Dirac BSDFs. TODO

                                // TODO check if wo is wo
                                auto main_bsdf_eta = def(0.f);// eta at previous main it
                                pipeline().surfaces().dispatch(previous_main_it.shape().surface_tag(), [&](auto surface) noexcept {
                                    auto closure = surface->closure(make_shared<Interaction>(previous_main_it), swl, -previous_main_ray.ray->direction(), 1.f, time);
                                    main_bsdf_eta = closure->eta().value_or(1.f);
                                });
                                auto shifted_bsdf_eta = def(0.f);// eta at previous main it
                                pipeline().surfaces().dispatch(shifted.it->shape().surface_tag(), [&](auto surface) noexcept {
                                    auto closure = surface->closure(shifted.it, swl, -shifted.ray.ray->direction(), 1.f, time);
                                    shifted_bsdf_eta = closure->eta().value_or(1.f);
                                });

                                auto main_tangent_space_wo = previous_main_it.shading().world_to_local(main_bsdf_result.wo);
                                auto main_tangent_space_wi = previous_main_it.shading().world_to_local(main_bsdf_result.sample.wi);
                                auto shift_result = half_vector_shift(
                                    main_tangent_space_wo, main_tangent_space_wi,
                                    tangent_space_incoming_direction,
                                    main_bsdf_eta, shifted_bsdf_eta);

                                // TODO  BSDF:EDelta

                                auto shift_failed_flag = def(false);
                                auto shifted_bsdf_pdf = def(0.f);
                                auto shifted_lum_pdf = def(0.f);
                                auto shifted_bsdf_value = SampledSpectrum{swl.dimension(), 0.f};
                                $if(shift_result.success) {
                                    shifted.pdf_div_main_pdf *= shift_result.jacobian;
                                    tangent_space_outgoing_direction = shift_result.wo;
                                }
                                $else {
                                    shifted.alive = false;
                                    shift_failed_flag = true;
                                };

                                auto outgoing_direction = shifted.it->shading().local_to_world(tangent_space_outgoing_direction);
                                $if(!shift_failed_flag) {
                                    pipeline().surfaces().dispatch(shifted.it->shape().surface_tag(), [&](auto surface) noexcept {
                                        auto closure = surface->closure(shifted.it, swl, -shifted.ray.ray->direction(), 1.f, time);
                                        // TODO check if tangent space is correct
                                        auto eval = closure->evaluate(-shifted.ray.ray->direction(), outgoing_direction);

                                        $if(eval.pdf <= 0.f) {
                                            // invalid path
                                            shifted.alive = false;
                                            shift_failed_flag = true;
                                        }
                                        $else {
                                            shifted_bsdf_pdf = eval.pdf;
                                            shifted_bsdf_value = eval.f;
                                            shifted.weight *= eval.f / eval.pdf;
                                            shifted.pdf_div_main_pdf *= eval.pdf / main_bsdf_pdf;
                                        };
                                    });
                                    // Strict normal TODO
                                };

                                $if(!shift_failed_flag) {
                                    auto shifted_vertex_type = get_vertex_type(shifted.it, swl, time);
                                    shifted.ray.ray = shifted.it->spawn_ray(outgoing_direction);
                                    *shifted.it = *pipeline().geometry()->intersect(shifted.ray.ray);

                                    $if(!shifted.it->valid()) {
                                        if (!pipeline().environment()) {
                                            shifted.alive = false;
                                            shift_failed_flag = true;
                                        } else {
                                            $if(main.it->valid()) {
                                                shifted.alive = false;
                                                shift_failed_flag = true;
                                            }
                                            $elif(main_vertex_type == (uint)VertexType::VERTEX_TYPE_DIFFUSE & shifted_vertex_type == (uint)VertexType::VERTEX_TYPE_DIFFUSE) {
                                                shifted.alive = false;
                                                shift_failed_flag = true;
                                            }
                                            $else {
                                                auto eval = light_sampler()->evaluate_miss(shifted.ray.ray->direction(), swl, time);
                                                shifted_emitter_radiance = eval.L;
                                                shifted_lum_pdf = eval.pdf;
                                                postponed_shift_end = true;
                                            };
                                        }
                                    }
                                    $else {
                                        $if(!main.it->valid()) {
                                            shifted.alive = false;
                                            shift_failed_flag = true;
                                        }
                                        $else {
                                            auto shifted_next_vertex_type = get_vertex_type(shifted.it, swl, time);
                                            $if(main_vertex_type == (uint)VertexType::VERTEX_TYPE_DIFFUSE &
                                                shifted_vertex_type == (uint)VertexType::VERTEX_TYPE_DIFFUSE &
                                                shifted_next_vertex_type == (uint)VertexType::VERTEX_TYPE_DIFFUSE) {
                                                shifted.alive = false;
                                                shift_failed_flag = true;
                                            }
                                            $else {
                                                $if(shifted.it->shape().has_light()) {
                                                    auto eval = light_sampler()->evaluate_hit(*shifted.it, shifted.ray.ray->origin(), swl, time);
                                                    shifted_lum_pdf = eval.pdf;
                                                    shifted_emitter_radiance = eval.L;
                                                };
                                                // TODO subsurface
                                            };
                                        };
                                    };
                                    // half vector shifted failed should go here
                                    $if(shifted.alive) {
                                        main_contribution = main.weight * main_emitter_radiance /
                                                            (1.f / balance_heuristic(main_bsdf_pdf, main_lum_pdf) + shifted.pdf_div_main_pdf / balance_heuristic(shifted_bsdf_pdf, shifted_lum_pdf));
                                        shifted_contribution = shifted.weight * shifted_emitter_radiance /
                                                               (1.f / balance_heuristic(main_bsdf_pdf, main_lum_pdf) / shifted.pdf_div_main_pdf + 1.f / balance_heuristic(shifted_bsdf_pdf, shifted_lum_pdf));
                                    }
                                    $else {
                                        main_contribution = main.weight * main_emitter_radiance * balance_heuristic(main_bsdf_pdf, main_lum_pdf);
                                        shifted_contribution = SampledSpectrum{swl.dimension(), 0.f};

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
                    main_contribution = main_weight * main_emitter_radiance * main_bsdf_result.weight;
                    shifted_contribution = SampledSpectrum{swl.dimension(), 0.f};
                };

                if (!node<GradientPathTracing>()->central_radiance()) {
                    main.add_radiance(main_contribution);
                    shifted.add_radiance(shifted_contribution);
                }
                shifted.add_gradient(shifted_contribution - main_contribution);

                shifted.alive = ite(postponed_shift_end, false, shifted.alive);
            }

            // Stop if the base path hit the environment.
            // TODO main.rRec.type
            $if(!main.it->valid() /*| !(main.rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance)*/) {
                $break;
            };

            $if(depth >= node<GradientPathTracing>()->rr_depth()) {
                // Russian Roulette
                auto threshold = node<GradientPathTracing>()->rr_threshold();
                auto q = max((main.weight).max(), 0.05f);
                $if(q < threshold & sampler()->generate_1d() >= q) {
                    $break;
                };

                $if(q < threshold) {
                    main.weight /= q;
                    for (int i = 0; i < 4; i++) {
                        shifteds[i].weight /= q;
                    }
                };
            };
        };
    };

    return result;
};

void GradientPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Camera::Instance *camera) noexcept {
    if (!pipeline().has_lighting()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "No lights in scene. Rendering aborted.");
        return;
    }

    auto spp = camera->node()->spp();
    auto resolution = camera->film()->node()->resolution();
    auto image_file = camera->node()->file();

    luisa::unordered_map<luisa::string, luisa::unique_ptr<ImageBuffer>> image_buffers;
    if (!node<GradientPathTracing>()->central_radiance()) {
        image_buffers.emplace("gradient_x", luisa::make_unique<ImageBuffer>(pipeline(), resolution));
        image_buffers.emplace("gradient_y", luisa::make_unique<ImageBuffer>(pipeline(), resolution));
    }

    auto clear_image_buffer = [&] {
        for (auto &[_, buffer] : image_buffers) {
            buffer->clear(command_buffer);
        }
    };

    auto pixel_count = resolution.x * resolution.y;
    sampler()->reset(command_buffer, resolution, pixel_count, spp);
    command_buffer << compute::synchronize();

    LUISA_INFO(
        "Rendering to '{}' of resolution {}x{} at {}spp.",
        image_file.string(),
        resolution.x, resolution.y, spp);

    using namespace luisa::compute;

    Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
        set_block_size(16u, 16u, 1u);
        auto pixel_id = dispatch_id().xy();
        float diff_scale_factor = 1.0f / std::sqrt((float)spp);
        auto eval = evaluate_point(pixel_id, frame_index, time, diff_scale_factor, camera);
        if (node<GradientPathTracing>()->central_radiance()) {
            auto L = pipeline().spectrum()->srgb(eval.swl, eval.very_direct + eval.throughput);
            camera->film()->accumulate(pixel_id, shutter_weight * L);
        } else {
            for (int i = 0; i < 4; i++) {
                auto current_pixel = pixel_id + pixel_shifts[i];
                $if(all(current_pixel >= 0u && current_pixel < resolution)) {
                    auto L = pipeline().spectrum()->srgb(eval.swl, 2.f * eval.neighbor_throughput[i]);
                    camera->film()->accumulate(current_pixel, shutter_weight * L, 1.f);
                };
            }
            auto L = pipeline().spectrum()->srgb(eval.swl, 8.f * eval.very_direct + 2.f * eval.throughput);
            camera->film()->accumulate(pixel_id, shutter_weight * L, 4.f);

            for (int i = 0; i < 4; i++) {
                auto current_pixel = pixel_id + pixel_shifts[i];
                auto key = i % 2 == 0 ? "gradient_x" : "gradient_y";
                auto sign = i < 2 ? 1.f : -1.f;
                $if(all(current_pixel >= 0u && current_pixel < resolution)) {
                    auto L = pipeline().spectrum()->srgb(eval.swl, sign * (2.f * eval.gradients[i] - eval.very_direct));
                    image_buffers.at(key)->accumulate(current_pixel, shutter_weight * L, 1.f);
                };
            }
        }
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
    clear_image_buffer();
    for (auto s : shutter_samples) {
        pipeline().update(command_buffer, s.point.time);
        for (auto i = 0u; i < s.spp; i++) {
            command_buffer << render(sample_id++, s.point.time, s.point.weight)
                                  .dispatch(resolution);
            if (auto &&p = pipeline().printer(); !p.empty()) {
                command_buffer << p.retrieve();
            }
            auto dispatches_per_commit =
                display() && !display()->should_close() ?
                    node<ProgressiveIntegrator>()->display_interval() :
                    32u;
            if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                dispatch_count = 0u;
                auto p = sample_id / static_cast<double>(spp);
                if (display() && display()->update(command_buffer, sample_id)) {
                    progress.update(p);
                } else {
                    command_buffer << [&progress, p] { progress.update(p); };
                }
            }
        }
    }
    auto parent_path = camera->node()->file().parent_path();
    auto filename = camera->node()->file().stem().string();
    auto ext = camera->node()->file().extension().string();
    command_buffer << synchronize();
    for (auto &[key, buffer] : image_buffers) {
        auto path = parent_path / fmt::format("{}_{}{}", filename, key, ext);
        command_buffer << buffer->save(command_buffer, path);
    }
    command_buffer << synchronize();
    progress.done();

    auto render_time = clock.toc();
    LUISA_INFO("Rendering finished in {} ms.", render_time);
}

[[nodiscard]] Float3 GradientPathTracingInstance::Li(const Camera::Instance *camera,
                                                     Expr<uint> frame_index,
                                                     Expr<uint2> pixel_id,
                                                     Expr<float> time) const noexcept {
    auto spp = camera->node()->spp();
    float diff_scale_factor = 1.0f / std::sqrt((float)spp);
    auto eval = evaluate_point(pixel_id, frame_index, time, diff_scale_factor, camera);

    return pipeline().spectrum()->srgb(eval.swl, eval.very_direct + eval.throughput);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GradientPathTracing)
