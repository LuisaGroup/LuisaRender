//
// Created by Hercier on 2023/3/4.
//

#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/display.h>

namespace luisa::render {

using namespace compute;
//Problem List:
//No fully support for environment due to const world radius
//Assumption: swl is fixed
class MegakernelPhotonMapping final : public ProgressiveIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _photon_per_iter;
    float _initial_radius;

public:
    MegakernelPhotonMapping(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ProgressiveIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 0u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _initial_radius{std::max(desc->property_float_or_default("initial_radius", .1f),.00001f},
          _photon_per_iter{std::max(desc->property_uint_or_default("photon_per_iter", 100000u), 10u)} {};
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto photon_per_iter() const noexcept { return _photon_per_iter; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto initial_radius() const noexcept { return _initial_radius; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelPhotonMappingInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;
    class PhotonMap {
    private:
        //Some problem:currently can only initialize for super large photon cache(max_depth*photon_per_iter)
        Buffer<uint> _grid_head;
        Buffer<float> _beta;
        Buffer<float3> _wi;
        Buffer<float3> _position;
        UInt _size;
        Buffer<uint> _tot;
        const Spectrum::Instance *_spectrum;
        UInt position_hash(Float3 position) {
        }

    public:
        PhotonMap(Device &device, uint photon_count, const Spectrum::Instance *spectrum) {
            _grid_head = device.create_buffer<uint>(photon_count);
            _beta = device.create_buffer<float>(photon_count * _spectrum->node()->dimension());
            _wi = device.create_buffer<float3>(photon_count);
            _position = device.create_buffer<float3>(photon_count);
            _tot = device.create_buffer<uint>(1u);
            _size = photon_count;
            _spectrum = spectrum;
        }
        void push(Expr<float3> position, SampledSpectrum power, Expr<float3> wi) {
            auto index = _tot.atomic(0).fetch_add(1u);
            auto dimension = _spectrum->node()->dimension();
            _wi.write(index, wi);
            _position.write(index, position);
            for (auto i = 0u; i < dimension; ++i)
                _beta.write(index * dimension + i, position);
        }

    } photons;
    class PixelIndirect {
        Buffer<float> _radius;
        Buffer<float> _cur_n;
        Buffer<float> _n_photon;
        Buffer<float> _phi;
        Buffer<float> _tau;
        const Film::Instance *_film;
        const Spectrum::Instance *_spectrum;

    public:
        PixelIndirect(Device &device, const Spectrum::Instance *spectrum, const Film::Instance *film) {
            _film = film;
            _spectrum = spectrum;
            auto resolution = film->node()->resolution();
            auto dimension = spectrum->node()->dimension();
            _radius = device.create_buffer<float>(resolution.x * resolution.y);
            _cur_n = device.create_buffer<float>(resolution.x * resolution.y);
            _n_photon = device.create_buffer<float>(resolution.x * resolution.y);
            _phi = device.create_buffer<float>(resolution.x * resolution.y * dimension);
            _tau = device.create_buffer<float>(resolution.x * resolution.y * dimension);
        }
        void write_radius(Expr<uint2> pixel_id, Expr<float> value) noexcept {
            auto resolution = _film->node()->resolution();
            _radius.write(pixel_id.y * resolution.x + pixel_id.x, value);
        }
        void write_cur_n(Expr<uint2> pixel_id, Expr<float> value) noexcept {
            auto resolution = _film->node()->resolution();
            _cur_n.write(pixel_id.y * resolution.x + pixel_id.x, value);
        }
        void write_n_photon(Expr<uint2> pixel_id, Expr<float> value) noexcept {
            auto resolution = _film->node()->resolution();
            _n_photon.write(pixel_id.y * resolution.x + pixel_id.x, value);
        }
        void reset_phi(Expr<uint2> pixel_id) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = _spectrum->node()->dimension();
            for (auto i = 0u; i < dimension; ++i)
                _phi.write(offset * dimension + i, 0.f);
        }
        void write_tau(Expr<uint2> pixel_id, Local<float> &value) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = _spectrum->node()->dimension();
            for (auto i = 0u; i < dimension; ++i)
                _phi.write(offset * dimension + i, value[i]);
        }
        Float radius(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            return _radius.read(pixel_id.y * resolution.x + pixel_id.x);
        }
        Float n_photon(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            return _n_photon.read(pixel_id.y * resolution.x + pixel_id.x);
        }
        Float cur_n(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            return _n_photon.read(pixel_id.y * resolution.x + pixel_id.x);
        }
        Local<float> phi(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = _spectrum->node()->dimension();
            Local<float> ret{dimension};
            for (auto i = 0u; i < dimension; ++i)
                ret[i] = _phi.read(offset * dimension + i);
            return ret;
        }
        Local<float> tau(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = _spectrum->node()->dimension();
            Local<float> ret{dimension};
            for (auto i = 0u; i < dimension; ++i)
                ret[i] = _tau.read(offset * dimension + i);
            return ret;
        }
        void add_cur_n(Expr<uint2> pixel_id, Expr<float> value) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            _cur_n.atomic(offset).fetch_add(value);
        }
        void add_phi(Expr<uint2> pixel_id, Local<float> &phi) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = _spectrum->node()->dimension();
            for (auto i = 0u; i < dimension; ++i)
                _phi.atomic(offset * dimension + i).fetch_add(phi[i]);
        }
    } indirect;

protected:
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
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
        const float WorldR = 100;
        auto &&device = camera->pipeline().device();
        photons = PhotonMap(device, node<MegakernelPhotonMapping>()->photon_per_iter() * node<MegakernelPhotonMapping>()->max_depth(), camera->pipeline().spectrum());
        indirect = PixelIndirect(device, camera->pipeline().spectrum(), camera->film());
        Kernel2D indirect_initialize_kernel = [&]() noexcept {

        };
        Kernel1D photon_reset_kernel = [&]() noexcept {

        };
        Kernel1D photon_grid_kernel = [&]() noexcept {

        };
        Kernel1D photon_emit_kernel = [&](UInt frame_index, Float time) noexcept {
            auto pixel_id = dispatch_id().xy();
            PhotonTracing(camera, frame_index, pixel_id, time);
        };
        Kernel2D render_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L = Li(camera, frame_index, pixel_id, time);
            camera->film()->accumulate(pixel_id, shutter_weight * L);
        };

        Kernel2D indirect_draw_kernel = [&](UInt tot_photon) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L = GetIndirect(pixel_id, tot_photon);
            camera->film()->accumulate(pixel_id, L, 0.f);
        };
        Kernel2D indirect_update_kernel = [&]() noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            PixelInfoUpdate(pixel_id);
        };
        Clock clock_compile;
        auto render = pipeline().device().compile(render_kernel);
        auto emit = pipeline().device().compile(photon_emit_kernel);
        auto update = pipeline().device().compile(indirect_update_kernel);
        auto indirect_draw = pipeline().device().compile(indirect_draw_kernel);
        auto integrator_shader_compilation_time = clock_compile.toc();
        auto indirect_initialize = pipeline().device().compile(indirect_initialize_kernel);
        auto indirect_update = pipeline().device().compile(indirect_update_kernel);
        auto photon_reset = pipeline().device().compile(photon_reset_kernel);
        auto photon_grid = pipeline().device().compile(photon_grid_kernel);
        LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
        auto shutter_samples = camera->node()->shutter_samples();
        command_buffer << synchronize();

        LUISA_INFO("Rendering started.");
        Clock clock;
        ProgressBar progress;
        progress.update(0.);
        auto dispatch_count = 0u;
        auto sample_id = 0u;
        command_buffer << indirect;
        command_buffer << synchronize();
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            for (auto i = 0u; i < s.spp; i++) {
                //emit phtons then calculate L
                //TODO: clear buffer
                command_buffer << emit(sample_id++, s.point.time, s.point.weight)
                                      .dispatch(resolution);
                command_buffer << synchronize();
                command_buffer << render(sample_id++, s.point.time)
                                      .dispatch(resolution);
                command_buffer << synchronize();
                command_buffer << update().dispatch(resolution);
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
        //command_buffer << synchronize();
        //camera->film()->convert(command_buffer);//normalize the spp
        command_buffer << synchronize();
        command_buffer << indirect_draw(node<MegakernelPhotonMapping>()->photon_per_iter()).dispatch(resolution);
        command_buffer << synchronize();
        progress.done();

        auto render_time = clock.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }
    [[nodiscard]] Float3 GetIndirect(const Spectrum::Instance *spectrum, Expr<uint2> pixel_id, Expr<uint> tot_photon) noexcept {
        auto r = indirect.radius(pixel_id);
        auto L = indirect.tau(pixel_id);
        for ()
            / (tot_photon * pi * r * r);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(0.f);
        return spectrum->srgb(swl, L);
    }
    void PixelInfoUpdate(Expr<uint2> pixel_id) {
        $if(indirect.cur_n(pixel_id) > 0) {
            Float n_new = indirect.n_photon(pixel_id) + 2.f / 3.f * indirect.cur_n(pixel_id);
            Float r_new = indirect.radius(pixel_id) * sqrt(n_new / (indirect.n_photon(pixel_id) + indirect.cur_n(pixel_id)));
            //indirect.write_tau(pixel_id, (indirect.tau(pixel_id) + indirect.phi(pixel_id)) * (r_new * r_new) / (indirect.radius(pixel_id) * indirect.radius(pixel_id)));
            indirect.write_tau(pixel_id, (r_new * r_new) / (indirect.radius(pixel_id) * indirect.radius(pixel_id)));
            indirect.write_n_photon(pixel_id, n_new);
            indirect.write_cur_n(pixel_id, 0.f);
            indirect.write_radius(pixel_id, r_new);
            indirect.reset_phi(pixel_id);
        };
    }
    [[nodiscard]] Float3 Li(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time) noexcept {
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
        $for(depth, node<MegakernelPhotonMapping>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
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

            // generate uniform samples
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<MegakernelPhotonMapping>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // sample one light
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape()->surface_tag();
            auto eta_scale = def(1.f);
            Bool stop_direct = false;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                auto closure = surface->closure(it, swl, wo, 1.f, time);

                // apply opacity map
                auto alpha_skip = def(false);
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                    pdf_bsdf = 1e16f;
                }
                $else {
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
                    //TODO: get this done
                    //if (closure->is_diffuse()) {
                    stop_direct = true;
                    //}
                };
            });
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            $if(stop_direct) {
                auto it_next = pipeline().geometry()->intersect(ray);

                // miss
                $if(!it_next->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    }
                };

                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it_next->shape()->has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it_next, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    };
                }
                //TODO:
                find_importon(it->p());
                $break;
            };
            auto rr_threshold = node<MegakernelPhotonMapping>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        return spectrum->srgb(swl, Li);
    }
    void PhotonTracing(const Camera::Instance *camera, Expr<uint> frame_index,
                       Expr<uint2> pixel_id, Expr<float> time) {

        sampler()->start(pixel_id, frame_index);
        // generate uniform samples
        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface = sampler()->generate_2d();
        auto u_direction = sampler()->generate_2d();
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto light_sample = light_sampler()->sample_le(
            u_light_selection, u_light_surface, u_direction, swl, time);
        //cos term include in L
        SampledSpectrum beta = light_sample.eval.L / light_sample.eval.pdf;

        auto ray = light_sample.shadow_ray;
        auto pdf_bsdf = def(1e16f);
        $for(depth, node<MegakernelPhotonMapping>()->max_depth()) {

            // trace
            auto wi = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            // miss
            $if(!it->valid()) {
                $break;
            };

            $if(!it->shape()->has_surface()) { $break; };

            // generate uniform samples
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<MegakernelPhotonMapping>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            $if(depth > 0){
                //find nearby importon and gives contribution
            };
            // evaluate material
            auto surface_tag = it->shape()->surface_tag();
            auto eta_scale = def(1.f);
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                // create closure
                //TODO: support for not bidirectional imp
                auto closure = surface->closure(it, swl, wi, 1.f, time);

                // apply opacity map
                auto alpha_skip = def(false);
                if (auto o = closure->opacity()) {
                    auto opacity = saturate(*o);
                    alpha_skip = u_lobe >= opacity;
                    u_lobe = ite(alpha_skip, (u_lobe - opacity) / (1.f - opacity), u_lobe / opacity);
                }

                $if(alpha_skip) {
                    ray = it->spawn_ray(ray->direction());
                    pdf_bsdf = 1e16f;
                }
                $else {
                    if (auto dispersive = closure->is_dispersive()) {
                        $if(*dispersive) { swl.terminate_secondary(); };
                    }

                    // sample material
                    auto surface_sample = closure->sample(wi, u_lobe, u_bsdf);
                    ray = it->spawn_ray(surface_sample.wi);
                    pdf_bsdf = surface_sample.eval.pdf;
                    auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                    //TODO: save photon
                    auto bnew = beta * w * surface_sample.eval.f;
                    // apply eta scale
                    auto eta = closure->eta().value_or(1.f);
                    $switch(surface_sample.event) {
                        $case(Surface::event_enter) { eta_scale = sqr(eta); };
                        $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                    };
                    eta_scale *= ite(beta.max() < bnew.max(), 1, bnew.max() / beta.max());
                    beta = bnew;
                };
            });
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            auto rr_threshold = node<MegakernelPhotonMapping>()->rr_threshold();
            auto q = max(eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
    }
};

luisa::unique_ptr<Integrator::Instance> MegakernelPhotonMapping::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelPhotonMappingInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPhotonMapping)
