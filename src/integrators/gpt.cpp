#include "EASTL/unique_ptr.h"
#include "base/geometry.h"
#include "base/interaction.h"
#include "base/spectrum.h"
#include "core/basic_traits.h"
#include "core/basic_types.h"
#include "core/mathematics.h"
#include "dsl/builtin.h"
#include "dsl/expr.h"
#include "dsl/var.h"
#include "rtx/ray.h"
#include "util/frame.h"
#include "util/scattering.h"
#include "util/spec.h"
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
    int m_max_depth;
	int m_min_depth;
    int m_rr_depth;
	bool m_strict_normals;
	float m_shift_threshold;
	bool m_reconstruct_L1;
	bool m_reconstruct_L2;
	float m_reconstruct_alpha;
};

enum VertexType {
    VERTEX_TYPE_GLOSSY,
    VERTEX_TYPE_DIFFUSE
};

enum RayConnection {
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
    Float eta;                      // Current refractive index
    Bool alive;
    RayConnection connection_status;

    RayState() : radiance(0.0f), gradient(0.0f), eta(1.0f), pdf(1.0f), 
                 throughput(0.0f), alive(true), connection_status(RAY_NOT_CONNECTED) {}

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

auto test_visibility(const Pipeline& pipeline, Expr<float3> point1, Expr<float3> point2) {
    auto shadow_ray = make_ray(
        point1, point2 - point1, epsilon, 1.f - shadow_epsilon
    );
    return !pipeline.geometry()->intersect_any(shadow_ray);
}

auto test_environment_visibility(const Pipeline& pipeline, const Var<Ray>& ray) {
    if(!pipeline.environment()) return def(false);
    auto shadow_ray = make_ray(
        ray->origin(), ray->direction(), epsilon, std::numeric_limits<float>::max()
    );
    // Miss = Intersect with env
    return !pipeline.geometry()->intersect_any(shadow_ray); 
}

auto get_vertex_type() noexcept { // no diffuse here
    return VertexType::VERTEX_TYPE_GLOSSY;
}

struct HalfVectorShiftResult {
    Bool success;
    Float jacobian;
    Float3 wo;
};

HalfVectorShiftResult half_vector_shift(
    Float3 tangent_space_main_wi,
    Float3 tangent_space_main_wo,
    Float3 tangent_space_shifted_wi,
    Float main_eta, Float shifted_eta) {
    HalfVectorShiftResult result;

    $if(cos_theta(tangent_space_main_wi) * cos_theta(tangent_space_shifted_wi) < 0.f) {
        // Refraction

        result.success = true;
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

ReconnectionShiftResult reconnect_shift(
    const Pipeline& pipeline, 
    Expr<float3> main_source_vertex, 
    Expr<float3> target_vertex,
    Expr<float3> shift_source_vertex,
    Expr<float3> target_normal) {
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

ReconnectionShiftResult environment_shift(
    const Pipeline& pipeline,
    const Var<Ray>& main_ray,
    Expr<float3> shift_source_vertex) {
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
};

void GradientPathTracingInstance::_render_one_camera(
    CommandBuffer &command_buffer, Camera::Instance *camera) noexcept {
    // auto spp = camera->node()->spp();
    // auto resolution = camera->film()->node()->resolution();
    // auto image_file = camera->node()->file();

    // auto pixel_count = resolution.x * resolution.y;
    // sampler()->reset(command_buffer, resolution, pixel_count, spp);
    // command_buffer << compute::synchronize();

    // LUISA_INFO(
    //     "Rendering to '{}' of resolution {}x{} at {}spp.",
    //     image_file.string(),
    //     resolution.x, resolution.y, spp);

    // using namespace luisa::compute;

    // Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
    //     set_block_size(16u, 16u, 1u);
    //     auto pixel_id = dispatch_id().xy();
    //     auto L = Li(camera, frame_index, pixel_id, time);
    //     camera->film()->accumulate(pixel_id, shutter_weight * L);
    // };

    // Clock clock_compile;
    // auto render = pipeline().device().compile(render_kernel);
    // auto integrator_shader_compilation_time = clock_compile.toc();
    // LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
    // auto shutter_samples = camera->node()->shutter_samples();
    // command_buffer << synchronize();

    // LUISA_INFO("Rendering started.");
    // Clock clock;
    // ProgressBar progress;
    // progress.update(0.);
    // auto dispatch_count = 0u;
    // auto sample_id = 0u;
    // for (auto s : shutter_samples) {
    //     pipeline().update(command_buffer, s.point.time);
    //     for (auto i = 0u; i < s.spp; i++) {
    //         command_buffer << render(sample_id++, s.point.time, s.point.weight)
    //                               .dispatch(resolution);
    //         auto dispatches_per_commit =
    //             _display && !_display->should_close() ?
    //                 node<ProgressiveIntegrator>()->display_interval() :
    //                 32u;
    //         if (++dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
    //             dispatch_count = 0u;
    //             auto p = sample_id / static_cast<double>(spp);
    //             if (_display && _display->update(command_buffer, sample_id)) {
    //                 progress.update(p);
    //             } else {
    //                 command_buffer << [&progress, p] { progress.update(p); };
    //             }
    //         }
    //     }
    // }
    // command_buffer << synchronize();
    // progress.done();

    // auto render_time = clock.toc();
    // LUISA_INFO("Rendering finished in {} ms.", render_time);    
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GradientPathTracing)
