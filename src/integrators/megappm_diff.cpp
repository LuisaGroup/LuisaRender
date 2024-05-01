//
// Created by Jiankai on 2024/3/4.
//

#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <sdl/scene_node_desc.h>

#include <limits>
#include <luisa-compute.h>

namespace luisa::render {

using namespace compute;
/* Procedure :
    1.emit photons and save them
    2.(first time only) initialize pixelinfo and get the proper initial radius based on emitted photons
    3.put photons in the hashmap grids
    4.render direct light seperately, stop at high roughness, find nearby 3*3*3 grids for photons and save the informations
    5.using shared(SPPM)/PPM update procedure for pixels
    6.if shared, a seperate update is performed, and the grid_len is also updated according to radius
    7.repeat until end, then draw the indirect light to film
*/
class MegakernelPhotonMappingDiff: public DifferentiableIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _photon_per_iter;
    float _initial_radius;
    bool _separate_direct;
    bool _shared_radius;
    int _debug_photon_id;
    uint _backward_iter;
    int _debug_mode;

public:
    MegakernelPhotonMappingDiff(Scene *scene, const SceneNodeDesc *desc) noexcept
        : DifferentiableIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 2u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _initial_radius{std::max(desc->property_float_or_default("initial_radius", -200.f), -10000.f)},//<0 for world_size/-radius (-grid count)
          _photon_per_iter{std::max(desc->property_uint_or_default("photon_per_iter", 200000u), 10u)},
          _debug_photon_id{desc->property_int_or_default("debug_photon_id", -1)},
          _debug_mode{desc->property_int_or_default("debug", 0)},
          _backward_iter{desc->property_uint_or_default("backward_iter", 1u)},
          _separate_direct{true}, //when false, use photon mapping for all flux and gathering at first intersection. Just for debug
          _shared_radius{true} {};//whether or not use the shared radius trick in SPPM paper. True is better in performance.
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto photon_per_iter() const noexcept { return _photon_per_iter; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto initial_radius() const noexcept { return _initial_radius; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] bool is_differentiable() const noexcept override { return true; }
    [[nodiscard]] auto separate_direct() const noexcept { return _separate_direct; }
    [[nodiscard]] auto shared_radius() const noexcept { return _shared_radius; }
    [[nodiscard]] auto debug_mode() const noexcept { return _debug_mode; }
    [[nodiscard]] auto debug_photon() const noexcept { return _debug_photon_id; }
    [[nodiscard]] auto backward_iter() const noexcept { return _backward_iter; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelPhotonMappingDiffInstance : public DifferentiableIntegrator::Instance {

public:
    using DifferentiableIntegrator::Instance::Instance;
    //A hash table for storing photons in grid
    //Some problem:can only initialize for super large photon cache(max_depth*photon_per_iter)
    //The fetchmax functions have wrong implementation in Luisa compute, so related feature are wrong now
    //(Including uint grid index, and inital_radius<0)
    class ViewPointMap {
    public:
        Buffer<uint> _grid_head;
        Buffer<float> _beta;
        Buffer<float3> _wo;
        Buffer<float3> _position;
        Buffer<uint> _pixel_id;
        Buffer<uint> _surface_tags;
        Buffer<uint> _nxt;
        uint _size;//current photon count
        const Spectrum::Instance *_spectrum;
        uint _dimension;
        Buffer<float> _grid_min;//atomic float3
        Buffer<float> _grid_max;//atomic float3
        Buffer<float> _grid_len;//the length of a single grid (float1)
        Buffer<float> _swl_lambda;
        Buffer<float> _swl_pdf;
        Buffer<uint> _tot;
        Buffer<uint> tot_test;
    public:
        ViewPointMap(uint viewpoint_count, const Spectrum::Instance *spectrum) {
            auto &&device = spectrum->pipeline().device();
            _grid_head = device.create_buffer<uint>(viewpoint_count);
            _beta = device.create_buffer<float>(viewpoint_count * spectrum->node()->dimension());
            _wo = device.create_buffer<float3>(viewpoint_count);
            _position = device.create_buffer<float3>(viewpoint_count);
            _pixel_id = device.create_buffer<uint>(viewpoint_count);
            _surface_tags = device.create_buffer<uint>(viewpoint_count);
            _nxt = device.create_buffer<uint>(viewpoint_count);
            _tot = device.create_buffer<uint>(1u);
            _grid_len = device.create_buffer<float>(1u);
            _grid_min = device.create_buffer<float>(3u);
            _grid_max = device.create_buffer<float>(3u);
            _size = viewpoint_count;
            _spectrum = spectrum;
            _dimension = 3;//spectrum->node()->dimension();
            if (!_spectrum->node()->is_fixed()) {
                _swl_lambda = device.create_buffer<float>(viewpoint_count * spectrum->node()->dimension());
                _swl_pdf = device.create_buffer<float>(viewpoint_count * spectrum->node()->dimension());
            }
            tot_test = device.create_buffer<uint>(1u);
        }
        auto tot_viewpoint() const noexcept {
            return _tot->read(0u);
        }
        auto grid_len() const noexcept {
            return _grid_len->read(0u);
        }
        auto size() const noexcept {
            return _size;
        }
        auto position(Expr<uint> index) const noexcept {
            return _position->read(index);
        }
        auto wo(Expr<uint> index) const noexcept {
            return _wo->read(index);
        }
        auto beta(Expr<uint> index) const noexcept {
            //auto dimension = _spectrum->node()->dimension();
            SampledSpectrum s{_dimension};
            for (auto i = 0u; i < _dimension; ++i)
                s[i] = _beta->read(index * _dimension + i);
            return s;
        }
        auto nxt(Expr<uint> index) const noexcept {
            return _nxt->read(index);
        }
        auto grid_head(Expr<uint> index) const noexcept {
            return _grid_head->read(index);
        }
        auto pixel_id(Expr<uint> index) const noexcept {
            return _pixel_id->read(index);
        }
        auto surface_tag(Expr<uint> index) const noexcept {
            return _surface_tags->read(index);
        }
        auto swl(Expr<uint> index) const noexcept {
            SampledWavelengths swl(_dimension);
            for (auto i = 0u; i < _dimension; ++i) {
                swl.set_lambda(i, _swl_lambda->read(index * _dimension + i));
                swl.set_pdf(i, _swl_pdf->read(index * _dimension + i));
            }
            return swl;
        }
        void push(Expr<float3> position, SampledWavelengths swl, SampledSpectrum power, Expr<float3> wi, Expr<uint> pixel_id, Expr<uint> surface_tag) {
            auto index = _tot->atomic(0u).fetch_add(1u);
            if (!_spectrum->node()->is_fixed()) {
                for (auto i = 0u; i < _dimension; ++i) {
                    _swl_lambda->write(index * _dimension + i, swl.lambda(i));
                    _swl_pdf->write(index * _dimension + i, swl.pdf(i));
                }
            }
            _wo->write(index, wi);
            _position->write(index, position);
            _pixel_id->write(index, pixel_id);
            _surface_tags->write(index, surface_tag);
            for (auto i = 0u; i < _dimension; ++i)
                _beta->write(index * _dimension + i, power[i]);
            for (auto i = 0u; i < 3u; ++i)
                _grid_min->atomic(i).fetch_min(position[i]);
            for (auto i = 0u; i < 3u; ++i)
                _grid_max->atomic(i).fetch_max(position[i]);
            _nxt->write(index, 0u);
        }
        //from uint3 grid id to hash index of the grid
        auto grid_to_index(Expr<int3> p) const noexcept {
            auto hash = ((p.x * 73856093) ^ (p.y * 19349663) ^
                         (p.z * 83492791)) %
                        (_size);
            return (hash + _size) % _size;
        }
        //from float3 position to uint3 grid id
        auto point_to_grid(Expr<float3> p) const noexcept {
            Float3 grid_min = {_grid_min->read(0),
                               _grid_min->read(1),
                               _grid_min->read(2)};
            return make_int3((p - grid_min) / grid_len()) + make_int3(2, 2, 2);
        }
        auto point_to_index(Expr<float3> p) const noexcept {
            return grid_to_index(point_to_grid(p));
        }

        void link(Expr<uint> index) {
            auto p = _position->read(index);
            auto grid_index = point_to_index(p);
            auto head = _grid_head->atomic(grid_index).exchange(index);
            _nxt->write(index, head);
        }

        void reset(Expr<uint> index) {
            _grid_head->write(index, ~0u);
            _tot->write(0, 0u);
            _nxt->write(index, ~0u);
            for (auto i = 0u; i < 3u; ++i) {
                _grid_min->write(i, std::numeric_limits<float>::max());
                _grid_max->write(i, -std::numeric_limits<float>::max());
            }
        }
        void write_grid_len(Expr<float> len) {
            _grid_len->write(0u, len);
        }
        auto split(Expr<float> grid_count) const noexcept {
            auto _grid_size = _spectrum->pipeline().geometry()->world_max() - _spectrum->pipeline().geometry()->world_min();
            return min(min(_grid_size.x / grid_count, _grid_size.y / grid_count), _grid_size.z / grid_count);
        }
    };
    //Store the information of pixel updates
    class PixelIndirect {
    public:
        Buffer<float> _radius;
        Buffer<uint> _cur_n;
        Buffer<float> _cur_w;
        Buffer<uint> _n_photon;
        Buffer<float> _phi;
        Buffer<float> _tau;
        Buffer<float> _weight;
        const Film::Instance *_film;
        const Spectrum::Instance *_spectrum;
        bool _shared_radius;
        uint _photon_per_iter;
        float _clamp;

    public:
        PixelIndirect(uint photon_per_iter, const Spectrum::Instance *spectrum, const Film::Instance *film, float clamp, bool shared_radius) {
            _film = film;
            _spectrum = spectrum;
            _clamp = clamp;
            auto device = spectrum->pipeline().device();
            auto resolution = film->node()->resolution();
            auto dimension = 3u;//always save rgb
            _shared_radius = shared_radius;
            if (shared_radius) {
                _radius = device.create_buffer<float>(1);
                _cur_n = device.create_buffer<uint>(1);
                _cur_w = device.create_buffer<float>(1);
                _n_photon = device.create_buffer<uint>(1);
            } else {
                _radius = device.create_buffer<float>(resolution.x * resolution.y);
                _cur_n = device.create_buffer<uint>(resolution.x * resolution.y);
                _cur_w = device.create_buffer<float>(resolution.x * resolution.y);
                _n_photon = device.create_buffer<uint>(resolution.x * resolution.y);
            }
            _phi = device.create_buffer<float>(resolution.x * resolution.y * dimension);
            _tau = device.create_buffer<float>(resolution.x * resolution.y * dimension);
            _weight = device.create_buffer<float>(resolution.x * resolution.y);
            _photon_per_iter = photon_per_iter;
        }
        void write_radius(Expr<uint> pixel_id, Expr<float> value) noexcept {
            if (!_shared_radius) {
                //auto resolution = _film->node()->resolution();
                _radius->write(pixel_id, value);
            } else {
                _radius->write(0u, value);
            }
        }
        void write_cur_n(Expr<uint> pixel_id, Expr<uint> value) noexcept {
            if (!_shared_radius) {
                //auto resolution = _film->node()->resolution();
                _cur_n->write(pixel_id, value);
            } else {
                _cur_n->write(0u, value);
            }
        }
        
        void write_cur_w(Expr<uint> pixel_id, Expr<float> value) noexcept {
            if (!_shared_radius) {
                //auto resolution = _film->node()->resolution();
                _cur_w->write(pixel_id, value);
            } else {
                _cur_w->write(0u, value);
            }
        }

        void write_n_photon(Expr<uint> pixel_id, Expr<uint> value) noexcept {
            if (!_shared_radius) {
                //auto resolution = _film->node()->resolution();
                _n_photon->write(pixel_id, value);
            } else {
                _n_photon->write(0u, value);
            }
        }

        void reset_phi(Expr<uint> pixel_id) noexcept {
            //auto resolution = _film->node()->resolution();
            //auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            for (auto i = 0u; i < dimension; ++i)
                _phi->write(pixel_id * dimension + i, 0.f);
        }

        void reset_tau(Expr<uint> pixel_id) noexcept {
            //auto resolution = _film->node()->resolution();
            //auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            for (auto i = 0u; i < dimension; ++i)
                _tau->write(pixel_id * dimension + i, 0.f);
        }

        auto radius(Expr<uint> pixel_id) const noexcept {
            if (!_shared_radius) {
                //auto resolution = _film->node()->resolution();
                return _radius->read(pixel_id);
            } else {
                return _radius->read(0u);
            }
        }

        //tau=(tau+clamp(phi))*value, see pixel_info_update for useage
        void update_tau(Expr<uint> pixel_id, Expr<float> value) noexcept {
            //auto resolution = _film->node()->resolution();
            //auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            auto thershold = _clamp;
            for (auto i = 0u; i < dimension; ++i) {
                auto old_tau = _tau->read(pixel_id * dimension + i);
                auto phi = _phi->read(pixel_id * dimension + i);
                phi = max(-thershold, min(phi, thershold));//-thershold for wavelength sampling
                _tau->write(pixel_id * dimension + i, (old_tau + phi) * value);
            }
        }

        auto cur_w(Expr<uint> pixel_id) const noexcept {
            if (!_shared_radius) {
                //auto resolution = _film->node()->resolution();
                return _cur_w->read(pixel_id);
            } else {
                return _cur_w->read(0u);
            }
        }
        //weight=(weight+clamp(cur_w))*value, see pixel_info_update for useage
        void update_weight(Expr<uint> pixel_id, Expr<float> value) noexcept {
            //auto resolution = _film->node()->resolution();
            //auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto old_weight = _weight->read(pixel_id);
            auto cur_w_ = cur_w(pixel_id);
            _weight->write(pixel_id, (old_weight + cur_w_) * value);
        }

        auto n_photon(Expr<uint> pixel_id) const noexcept {
            //auto resolution = _film->node()->resolution();
            if (!_shared_radius) {
                return _n_photon->read(pixel_id);
            } else {
                return _n_photon->read(0u);
            }
        }

        auto cur_n(Expr<uint> pixel_id) const noexcept {
            if (!_shared_radius) {
                //auto resolution = _film->node()->resolution();
                return _cur_n->read(pixel_id);
            } else {
                return _cur_n->read(0u);
            }
        }
        
        auto weight(Expr<uint> pixel_id) const noexcept {
            return _weight->read(pixel_id);
        }

        auto phi(Expr<uint> pixel_id) const noexcept {
            //auto resolution = _film->node()->resolution();
            //auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            Float3 ret;
            for (auto i = 0u; i < dimension; ++i)
                ret[i] = _phi->read(pixel_id * dimension + i);
            return ret;
        }
        
        auto tau(Expr<uint> pixel_id) const noexcept {
            //auto resolution = _film->node()->resolution();
            //auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            Float3 ret;
            for (auto i = 0u; i < dimension; ++i)
                ret[i] = _tau->read(pixel_id * dimension + i);
            return ret;
        }

        void add_cur_n(Expr<uint> pixel_id, Expr<uint> value) noexcept {
            if (!_shared_radius) {
                //auto resolution = _film->node()->resolution();
                //auto offset = pixel_id.y * resolution.x + pixel_id.x;
                _cur_n->atomic(pixel_id).fetch_add(value);
            } else {
                _cur_n->atomic(0u).fetch_add(value);
            }
        }

        void add_cur_w(Expr<uint> pixel_id, Expr<float> value) noexcept {
            if (!_shared_radius) {
                //auto resolution = _film->node()->resolution();
                //auto offset = pixel_id.y * resolution.x + pixel_id.x;
                _cur_w->atomic(pixel_id).fetch_add(value);
            } else {
                _cur_w->atomic(0u).fetch_add(value);
            }
        }

        void add_phi(Expr<uint> pixel_id, Expr<float3> phi) noexcept {
            //auto resolution = _film->node()->resolution();
            //auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            for (auto i = 0u; i < dimension; ++i)
                _phi->atomic(pixel_id * dimension + i).fetch_add(phi[i]);
        }

        void pixel_info_update(Expr<uint> pixel_id) {
            $if(cur_n(pixel_id) > 0) {
                Float gamma = 2.0f / 3.0f;
                UInt n_new = n_photon(pixel_id) + cur_n(pixel_id);
                Float r_new = radius(pixel_id) * sqrt(n_new * gamma / (n_photon(pixel_id) * gamma + cur_n(pixel_id)));
                //indirect->write_tau(pixel_id, (indirect->tau(pixel_id) + indirect->phi(pixel_id)) * (r_new * r_new) / (indirect->radius(pixel_id) * indirect->radius(pixel_id)));
                update_tau(pixel_id, r_new * r_new / (radius(pixel_id) * radius(pixel_id)));
                update_weight(pixel_id, r_new * r_new / (radius(pixel_id) * radius(pixel_id)));
                if (!_shared_radius) {
                    write_n_photon(pixel_id, n_new);
                    write_cur_n(pixel_id, 0u);
                    write_radius(pixel_id, r_new);
                }
                reset_phi(pixel_id);
            };
        }
        void shared_update() {
            auto pixel_id = 0u;
            $if(cur_n(pixel_id) > 0) {
                Float gamma = 2.0f / 3.0f;
                UInt n_new = n_photon(pixel_id) + cur_n(pixel_id);
                Float r_new = radius(pixel_id) * sqrt(n_new * gamma / (n_photon(pixel_id) * gamma + cur_n(pixel_id)));
                write_n_photon(pixel_id, n_new);
                write_cur_n(pixel_id, 0u);
                write_radius(pixel_id, r_new);
            };
        }
    };

    luisa::unique_ptr<ViewPointMap> viewpoints;
    luisa::unique_ptr<PixelIndirect> indirect;
    const UInt max_size = 4u;
    const UInt param_size_per_vert = 5u;
    const UInt _grad_dimension = 5u;

protected:
    void _render_one_camera_backward(CommandBuffer &command_buffer, uint iteration,  Camera::Instance *camera, Buffer<float> &grad_in) noexcept { 
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        auto spp = camera->node()->spp();
        auto resolution = camera->film()->node()->resolution();
        auto image_file = camera->node()->file();
        auto photon_per_iter = node<MegakernelPhotonMappingDiff>()->photon_per_iter();
        auto pixel_count = resolution.x * resolution.y;
        auto spectrum = camera->pipeline().spectrum();
        
        uint add_x = (photon_per_iter + resolution.y - 1) / resolution.y;
        sampler()->reset(command_buffer, make_uint2(resolution.x + add_x, resolution.y), pixel_count + add_x * resolution.y, spp);
        
        command_buffer << pipeline().printer().reset();
        command_buffer << compute::synchronize();
        LUISA_INFO(
            "Rendering to '{}' of resolution {}x{} at {}spp.",
            image_file.string(),
            resolution.x, resolution.y, spp);

        using namespace luisa::compute;
        auto &&device = camera->pipeline().device();
        auto radius = node<MegakernelPhotonMappingDiff>()->initial_radius();
        if (radius < 0) {
            auto _grid_size = spectrum->pipeline().geometry()->world_max() - spectrum->pipeline().geometry()->world_min();
            radius = min(min(_grid_size.x / -radius, _grid_size.y / -radius), _grid_size.z / -radius);
        }
        auto clamp = camera->film()->node()->clamp() * photon_per_iter * pi * radius * radius;

        auto viewpoints_per_iter = resolution.x * resolution.y;
        
        indirect = make_unique<PixelIndirect>(viewpoints_per_iter, spectrum, camera->film(), clamp, node<MegakernelPhotonMappingDiff>()->shared_radius());
        viewpoints = make_unique<ViewPointMap>(viewpoints_per_iter, spectrum);
        
        Kernel1D indirect_initialize_kernel = [&]() noexcept {
            
            auto index = dispatch_x();
            auto radius = node<MegakernelPhotonMappingDiff>()->initial_radius();
            if (radius < 0)
                viewpoints->write_grid_len(viewpoints->split(-radius));
            else
                viewpoints->write_grid_len(node<MegakernelPhotonMappingDiff>()->initial_radius());
            //camera->pipeline().printer().info("grid:{}", viewpoints->grid_len());
            indirect->write_radius(index, viewpoints->grid_len());
            //camera->pipeline().printer().info("rad:{}", indirect->radius(index));

            indirect->write_cur_n(index, 0u);
            indirect->write_cur_w(index, 0.f);
            indirect->write_n_photon(index, 0u);
            indirect->reset_phi(index);
            indirect->reset_tau(index);
        };
        
        Kernel1D viewpoint_reset_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            viewpoints->reset(index);
        };

        Kernel2D viewpath_construct_kernel = [&](UInt frame_index, Float time, Float shutter_weight, BufferFloat grad_in) noexcept {
            //Construct view path
            auto pixel_id = dispatch_id().xy();
            auto L = emit_viewpoint_bp(camera, frame_index, pixel_id, time, shutter_weight, grad_in);
            $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 2) % 2 == 0 ) {
                $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 16) % 2 == 1 ){
                    camera->film()->accumulate(pixel_id, L, 0.5f);
                };
            };
        };

        Kernel1D build_grid_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            $if(viewpoints->nxt(index) == 0u) {
                viewpoints->link(index);
            };
        };

        Kernel2D emit_photons_kernel = [&](UInt frame_index, Float time, BufferFloat grad_in) noexcept {
            auto pixel_id = dispatch_id().xy();
            auto sampler_id = UInt2(pixel_id.x + resolution.x, pixel_id.y);
            $if(pixel_id.y * resolution.x + pixel_id.x < photon_per_iter) {
                photon_tracing_bp(camera, frame_index, sampler_id, time, pixel_id.y * resolution.x + pixel_id.x, grad_in);
            };
        };

        Kernel2D indirect_draw_kernel = [&](UInt tot_photon, UInt spp) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto pixel_id_1d = pixel_id.y * resolution.x + pixel_id.x;
            auto L = get_indirect(camera->pipeline().spectrum(), pixel_id_1d, tot_photon);
            camera->film()->accumulate(pixel_id, L, 0.5f * spp);
        };

        Kernel1D indirect_update_kernel = [&]() noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_x();
            indirect->pixel_info_update(pixel_id);
        };

        Kernel1D shared_update_kernel = [&]() noexcept {
            indirect->shared_update();
            viewpoints->write_grid_len(indirect->radius(0u));
        };

        Clock clock_compile;
        auto indirect_initialize = pipeline().device().compile(indirect_initialize_kernel);
        auto viewpoint_reset = pipeline().device().compile(viewpoint_reset_kernel);
        auto viewpath_construct = pipeline().device().compile(viewpath_construct_kernel);
        auto build_grid = pipeline().device().compile(build_grid_kernel);
        auto emit_photon = pipeline().device().compile(emit_photons_kernel);
        auto indirect_draw = pipeline().device().compile(indirect_draw_kernel);
        auto indirect_update = pipeline().device().compile(indirect_update_kernel);
        auto shared_update = pipeline().device().compile(shared_update_kernel);

        auto integrator_shader_compilation_time = clock_compile.toc();
        LUISA_INFO("Integrator shader compile in {} ms.", integrator_shader_compilation_time);
        auto shutter_samples = camera->node()->shutter_samples();
        command_buffer << synchronize();

        LUISA_INFO("Backward Rendering started.");
        Clock clock;
        ProgressBar progress;
        progress.update(0.);
        auto dispatch_count = 0u;
        auto sample_id = 0u;
        bool initial_flag = false;
        uint runtime_spp = 0u;

        command_buffer << indirect_initialize().dispatch(viewpoints_per_iter) << synchronize();
        pipeline().update(command_buffer, 0);

        for (auto s : shutter_samples) {
            runtime_spp+=node<MegakernelPhotonMappingDiff>()->backward_iter();
            for (auto i = 0u; i < node<MegakernelPhotonMappingDiff>()->backward_iter(); i++) {
                command_buffer << viewpoint_reset().dispatch(viewpoints->size());
                command_buffer << viewpath_construct(sample_id++, s.point.time, s.point.weight, grad_in).dispatch(resolution);
                command_buffer << build_grid().dispatch(viewpoints->size());
                command_buffer << emit_photon(sample_id++, s.point.time, grad_in).dispatch(make_uint2(add_x, resolution.y));
                command_buffer << indirect_update().dispatch(viewpoints_per_iter);
                if (node<MegakernelPhotonMappingDiff>()->shared_radius()) {
                    command_buffer << shared_update().dispatch(1u);
                }
            }
        }
        
        command_buffer << synchronize();
        // LUISA_INFO("Backward Rendering Forward Finished");
        // $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 16) % 2 == 1) {
        //     command_buffer << indirect_draw(node<MegakernelPhotonMappingDiff>()->photon_per_iter(), runtime_spp).dispatch(resolution);
        //     command_buffer << synchronize();
        //     LUISA_INFO("Backward Rendering Indirect Finished");
        // };
        progress.done();
        auto render_time = clock.toc();
        LUISA_INFO("Backward Rendering finished in {} ms.", render_time);
    }

    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
        
        if ((node<MegakernelPhotonMappingDiff>()->debug_mode() / 4) % 2 == 1) {
            Buffer<float> grad_in = pipeline().device().create_buffer<float>(camera->film()->node()->resolution().x * camera->film()->node()->resolution().y * 3);
            _render_one_camera_backward(command_buffer, 0, camera, grad_in);
            return;
        }
        
        if (!pipeline().has_lighting()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "No lights in scene. Rendering aborted.");
            return;
        }
        auto spp = camera->node()->spp();
        auto resolution = camera->film()->node()->resolution();
        auto image_file = camera->node()->file();
        auto photon_per_iter = node<MegakernelPhotonMappingDiff>()->photon_per_iter();
        auto pixel_count = resolution.x * resolution.y;
        auto spectrum = camera->pipeline().spectrum();
        
        uint add_x = (photon_per_iter + resolution.y - 1) / resolution.y;
        sampler()->reset(command_buffer, make_uint2(resolution.x + add_x, resolution.y), pixel_count + add_x * resolution.y, spp);
        
        command_buffer << pipeline().printer().reset();
        command_buffer << compute::synchronize();
        LUISA_INFO(
            "Rendering to '{}' of resolution {}x{} at {}spp.",
            image_file.string(),
            resolution.x, resolution.y, spp);

        using namespace luisa::compute;
        auto &&device = camera->pipeline().device();
        auto radius = node<MegakernelPhotonMappingDiff>()->initial_radius();
        if (radius < 0) {
            auto _grid_size = spectrum->pipeline().geometry()->world_max() - spectrum->pipeline().geometry()->world_min();
            radius = min(min(_grid_size.x / -radius, _grid_size.y / -radius), _grid_size.z / -radius);
        }
        auto clamp = camera->film()->node()->clamp() * photon_per_iter * pi * radius * radius;

        auto viewpoints_per_iter = resolution.x * resolution.y;
        
        indirect = make_unique<PixelIndirect>(viewpoints_per_iter, spectrum, camera->film(), clamp, node<MegakernelPhotonMappingDiff>()->shared_radius());
        viewpoints = make_unique<ViewPointMap>(viewpoints_per_iter, spectrum);
        //return;
        //pathlogger = make_unique<PathLogger>(node<MegakernelPhotonMappingDiff>()->max_depth(), node<MegakernelPhotonMappingDiff>()->photon_per_iter(), spectrum);
        //initialize PixelIndirect

        Kernel1D indirect_initialize_kernel = [&]() noexcept {
            
            auto index = dispatch_x();
            auto radius = node<MegakernelPhotonMappingDiff>()->initial_radius();
            if (radius < 0)
                viewpoints->write_grid_len(viewpoints->split(-radius));
            else
                viewpoints->write_grid_len(node<MegakernelPhotonMappingDiff>()->initial_radius());
            //camera->pipeline().printer().info("grid:{}", viewpoints->grid_len());
            indirect->write_radius(index, viewpoints->grid_len());
            //camera->pipeline().printer().info("rad:{}", indirect->radius(index));

            indirect->write_cur_n(index, 0u);
            indirect->write_cur_w(index, 0.f);
            indirect->write_n_photon(index, 0u);
            indirect->reset_phi(index);
            indirect->reset_tau(index);
        };
        
        Kernel1D viewpoint_reset_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            viewpoints->reset(index);
        };

        Kernel2D viewpath_construct_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            //Construct view path
            auto pixel_id = dispatch_id().xy();
            auto L = emit_viewpoint(camera, frame_index, pixel_id, time, shutter_weight);
            camera->film()->accumulate(pixel_id, L, 0.5f);
        };

        Kernel1D build_grid_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            auto radius = node<MegakernelPhotonMappingDiff>()->initial_radius();
            $if(viewpoints->nxt(index) == 0u) {
                viewpoints->link(index);
            };
        };

        Kernel2D emit_photons_kernel = [&](UInt frame_index, Float time) noexcept {
            auto pixel_id = dispatch_id().xy();
            auto sampler_id = UInt2(pixel_id.x + resolution.x, pixel_id.y);
            $if(pixel_id.y * resolution.x + pixel_id.x < photon_per_iter) {
                photon_tracing(camera, frame_index, sampler_id, time, pixel_id.y * resolution.x + pixel_id.x);
            };
        };

        Kernel1D indirect_update_kernel = [&]() noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_x();
            indirect->pixel_info_update(pixel_id);
        };

        Kernel1D shared_update_kernel = [&]() noexcept {
            indirect->shared_update();
            viewpoints->write_grid_len(indirect->radius(0u));
        };

        //accumulate the stored indirect light into final image
        Kernel2D indirect_draw_kernel = [&](UInt tot_photon, UInt spp) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto pixel_id_1d = pixel_id.y * resolution.x + pixel_id.x;
            // $if(pixel_id_1d == 0) {
            //     auto cur_n_tot = indirect->cur_n(0u);
            //     pipeline().printer().info("photon cur n is {}", cur_n_tot);
            // };
            auto L = get_indirect(camera->pipeline().spectrum(), pixel_id_1d, tot_photon);
            camera->film()->accumulate(pixel_id, L, 0.5f * spp);
        };
        Clock clock_compile;

        auto indirect_initialize = pipeline().device().compile(indirect_initialize_kernel);
        auto viewpoint_reset = pipeline().device().compile(viewpoint_reset_kernel);
        auto viewpath_construct = pipeline().device().compile(viewpath_construct_kernel);
        auto build_grid = pipeline().device().compile(build_grid_kernel);
        auto emit_photon = pipeline().device().compile(emit_photons_kernel);

        auto indirect_draw = pipeline().device().compile(indirect_draw_kernel);
        auto indirect_update = pipeline().device().compile(indirect_update_kernel);
        auto shared_update = pipeline().device().compile(shared_update_kernel);

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
        bool initial_flag = false;
        uint runtime_spp = 0u;
        

        command_buffer << indirect_initialize().dispatch(viewpoints_per_iter) << synchronize();
        pipeline().update(command_buffer, 0);
        for (auto s : shutter_samples) {
            runtime_spp+=spp;
            for (auto i = 0u; i < s.spp; i++) {
                command_buffer << viewpoint_reset().dispatch(viewpoints->size());
                command_buffer << viewpath_construct(sample_id++, s.point.time, s.point.weight).dispatch(resolution);
                command_buffer << build_grid().dispatch(viewpoints->size());
                command_buffer << emit_photon(sample_id++, s.point.time).dispatch(make_uint2(add_x, resolution.y));
                command_buffer << indirect_update().dispatch(viewpoints_per_iter);
                if (node<MegakernelPhotonMappingDiff>()->shared_radius()) {
                    command_buffer << shared_update().dispatch(1u);
                }
            }
        }

        command_buffer << synchronize();
        command_buffer << indirect_draw(node<MegakernelPhotonMappingDiff>()->photon_per_iter(), runtime_spp).dispatch(resolution);
        command_buffer << synchronize();
        command_buffer << pipeline().printer().retrieve();
        //LUISA_INFO("Finishi indirect_draw");
        progress.done();
        auto render_time = clock.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }

    [[nodiscard]] Float3 get_indirect(const Spectrum::Instance *spectrum, Expr<uint> pixel_id, Expr<uint> tot_photon) noexcept {
        auto r = indirect->radius(pixel_id);
        auto tau = indirect->tau(pixel_id);
        Float3 L = tau / (tot_photon * pi * r * r);
        return L;
    }

    [[nodiscard]] Float3 emit_viewpoint(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time, Expr<float> shutter_weight) noexcept {
        
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, _, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), shutter_weight * camera_weight};
        SampledSpectrum Li{swl.dimension()};
        SampledSpectrum testbeta{swl.dimension()};
        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);
        auto resolution = camera->film()->node()->resolution();
        auto pixel_id_1d = pixel_id.y*resolution.x+pixel_id.x;

        $for(depth, node<MegakernelPhotonMappingDiff>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                };
            }
            
            $if(!it->shape().has_surface()) { $break; };

            // generate uniform samples
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<MegakernelPhotonMappingDiff>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // sample one light
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            Bool stop_direct = false;
            auto rr_threshold = node<MegakernelPhotonMappingDiff>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);

            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { stop_direct = true; };
            };
            
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

                auto roughness = closure->roughness();
                Bool stop_check = (roughness.x * roughness.y > 0.16f) | stop_direct;
                $if(stop_check) {
                    stop_direct = true;
                    viewpoints->push(it->p(), swl, beta, wo, pixel_id_1d, surface_tag);
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
            $if(stop_direct) {
                auto it_next = pipeline().geometry()->intersect(ray);
                // logger->add_indirect_end(pixel_id_1d, path_size, beta);
                // miss
                $if(!it_next->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    }
                };
                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it_next->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it_next, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    };
                }
                $break;
            };
            $if(depth + 1u >= rr_depth) {
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        return spectrum->srgb(swl, Li);
    }
    
    [[nodiscard]] void photon_tracing(const Camera::Instance *camera, Expr<uint> frame_index,
                        Expr<uint2> sampler_id, Expr<float> time, Expr<uint> photon_id_1d) {
        sampler()->start(sampler_id, frame_index);
        // generate uniform samples
        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface = sampler()->generate_2d();
        auto u_direction = sampler()->generate_2d();
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto light_sample = light_sampler()->sample_le(
            u_light_selection, u_light_surface, u_direction, swl, time);
        //cos term canceled out in pdf
        SampledSpectrum beta = light_sample.eval.L / light_sample.eval.pdf;
        
        auto ray = light_sample.shadow_ray;
        auto pdf_bsdf = def(1e16f);

        $for(depth, node<MegakernelPhotonMappingDiff>()->max_depth()) {
            // trace
            auto wi = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);
            
            // miss
            $if(!it->valid()) {
                $break;
            };

            $if(!it->shape().has_surface()) { $break; };
            // generate uniform samples
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<MegakernelPhotonMappingDiff>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            $if(depth > 0) {
                auto grid = viewpoints->point_to_grid(it->p());
                $for(x, grid.x - 1, grid.x + 2) {
                    $for(y, grid.y - 1, grid.y + 2) {
                        $for(z, grid.z - 1, grid.z + 2) {
                            Int3 check_grid{x, y, z};
                            auto viewpoint_index = viewpoints->grid_head(viewpoints->grid_to_index(check_grid));
                            $while(viewpoint_index != ~0u) {
                                auto position = viewpoints->position(viewpoint_index);
                                auto pixel_id = viewpoints->pixel_id(viewpoint_index);
                                auto dis = distance(position, it->p());
                                
                                $if(dis <= indirect->radius(pixel_id)) {
                                    auto viewpoint_wo = viewpoints->wo(viewpoint_index);
                                    auto viewpoint_beta = viewpoints->beta(viewpoint_index);
                                    auto surface_tag = viewpoints->surface_tag(viewpoint_index);
                                    //auto eval_viewpoint = closure->evaluate(wi, viewpoint_wo);
                                    SampledSpectrum eval_viewpoint(3u);
                                    PolymorphicCall<Surface::Closure> call;
                                    pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                                        surface->closure(call, *it, swl, viewpoint_wo, 1.f, time);
                                    });
                                    call.execute([&](const Surface::Closure *closure) noexcept {
                                        eval_viewpoint = closure->evaluate(viewpoint_wo, wi).f;
                                    });
                                    auto wi_local = it->shading().world_to_local(wi);
                                    Float3 Phi;
                                    auto rel_dis = dis / indirect->radius(pixel_id);
                                    auto weight = 3.5f*(1.0f- 6 * pow(rel_dis, 5.) + 15 * pow(rel_dis, 4.) - 10 * pow(rel_dis, 3.));
                                    if (!spectrum->node()->is_fixed()) {
                                        auto viewpoint_swl = viewpoints->swl(pixel_id);
                                        Phi = spectrum->wavelength_mul(swl,  beta * (eval_viewpoint / abs_cos_theta(wi_local)), viewpoint_swl, viewpoint_beta);
                                    } else {
                                        Phi = spectrum->srgb(swl,  beta * viewpoint_beta * eval_viewpoint / abs_cos_theta(wi_local));
                                    }
                                    indirect->add_phi(pixel_id, Phi*weight);
                                    indirect->add_cur_n(pixel_id, 1u);
                                    //pipeline().printer().info("working here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! pixel_id and phi are {}, {} ", pixel_id, Phi);
                                };
                                viewpoint_index = viewpoints->nxt(viewpoint_index);
                                //pipeline().printer().info("working here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! pixel_id is ", pixel_id);
                            };
                        };
                    };
                };
            };

            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);

            PolymorphicCall<Surface::Closure> call;

            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wi, 1.f, time);
            });

            call.execute([&](auto closure) noexcept {
                // apply opacity map
                if (auto dispersive = closure->is_dispersive()) {
                    $if(*dispersive) { swl.terminate_secondary(); };
                }
                // sample material
                auto surface_sample = closure->sample(wi, u_lobe, u_bsdf, TransportMode::IMPORTANCE);
                ray = it->spawn_ray(surface_sample.wi);
                pdf_bsdf = surface_sample.eval.pdf;
                auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                auto bnew = beta * w * surface_sample.eval.f;
                // apply eta scale
                auto eta = closure->eta().value_or(1.f);
                $switch(surface_sample.event) {
                    $case(Surface::event_enter) { eta_scale = sqr(eta); };
                    $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); };
                };
                eta_scale *= ite(beta.max() < bnew.max(), 1.f, bnew.max() / beta.max());
                beta = bnew;
            });
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };
            auto rr_threshold = node<MegakernelPhotonMappingDiff>()->rr_threshold();
            auto q = max(eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { $break; };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
    }

    [[nodiscard]] Float3 emit_viewpoint_bp(const Camera::Instance *camera, Expr<uint> frame_index,
                            Expr<uint2> pixel_id, Expr<float> time, Expr<float> shutter_weight, BufferFloat &grad_in) noexcept {
        
        sampler()->start(pixel_id, frame_index);
        auto u_filter = sampler()->generate_pixel_2d();
        auto u_lens = camera->node()->requires_lens_sampling() ? sampler()->generate_2d() : make_float2(.5f);
        auto [camera_ray, pixel, camera_weight] = camera->generate_ray(pixel_id, time, u_filter, u_lens);
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        SampledSpectrum beta{swl.dimension(), shutter_weight * camera_weight};
        SampledSpectrum Li{swl.dimension()};
        SampledSpectrum testbeta{swl.dimension()};
        auto ray = camera_ray;
        auto pdf_bsdf = def(1e16f);

        auto resolution = camera->film()->node()->resolution();
        auto pixel_id_1d = pixel_id.y*resolution.x+pixel_id.x;

        Float3 start_point = camera_ray->origin();
        Float3 end_point;
        Float3 pixel_world;
        ArrayUInt<4> triangle_ids, inst_ids;
        ArrayFloat3<4> bary_coords, points, normals;
        ArrayFloat<4> etas;
        ArrayFloat2<4> grad_barys;
        ArrayFloat3<4> grad_betas;
        Bool flag = false;
        
        Float3 grad_rgb = make_float3(grad_in->read(pixel_id_1d*_grad_dimension), grad_in->read(pixel_id_1d*_grad_dimension+1), grad_in->read(pixel_id_1d*_grad_dimension+2));
        Float2 grad_pixel = make_float2(grad_in->read(pixel_id_1d*_grad_dimension+3), grad_in->read(pixel_id_1d*_grad_dimension+4));
        Float3 radiance = make_float3(1.0f, 1.0f, 1.0f);// = get_radiance_cache(pixel_id, frame_index);

        auto grad_pixel_world = make_float3(camera->camera_to_world() * make_float4(grad_pixel, 0.f, 1.f));
        auto path_size = 0u;

        $for(depth, node<MegakernelPhotonMappingDiff>()->max_depth()) {
            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);
            
            $if(!it->shape().has_surface()) { $break; };
            
            $if(depth == 0) {
                $autodiff {
                    auto p = it->p();
                    requires_grad(p);
                    auto pixel_pos_world = (p - start_point) * detach(distance(pixel_world, start_point) / distance(p, start_point));
                    backward(dot(pixel_pos_world, grad_pixel_world));
                    grad_barys[0] = get_bary_grad(grad(p), it->instance_id(), it->triangle_id());
                };
            };
            
            // generate uniform samples
            auto u_light_selection = sampler()->generate_1d();
            auto u_light_surface = sampler()->generate_2d();
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<MegakernelPhotonMappingDiff>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };

            // sample one light
            auto light_sample = light_sampler()->sample(
                *it, u_light_selection, u_light_surface, swl, time);

            // trace shadow ray
            auto occluded = pipeline().geometry()->intersect_any(light_sample.shadow_ray);

            // evaluate material
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            Bool stop_direct = false;
            auto rr_threshold = node<MegakernelPhotonMappingDiff>()->rr_threshold();
            auto q = max(beta.max() * eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) { stop_direct = true; };
            };

            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wo, 1.f, time);
            });

            call.execute([&](auto closure) noexcept {

                auto roughness = closure->roughness();
                Bool stop_check = (roughness.x * roughness.y > 0.16f) | stop_direct;
                $if(stop_check) {
                    stop_direct = true;
                    viewpoints->push(it->p(), swl, beta, wo, pixel_id_1d, surface_tag);
                };

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
                    $case(Surface::event_enter) { eta_scale = sqr(eta); etas[path_size] = eta;};
                    $case(Surface::event_exit) { eta_scale = sqr(1.f / eta); etas[path_size] = 1.f/eta;};
                };

                etas[path_size] = eta;
                points[path_size] = it->p();
                triangle_ids[path_size] = it->triangle_id();
                inst_ids[path_size] = it->instance_id();
                bary_coords[path_size] = it->bary_coord();
                points[path_size] = it->p();
                normals[path_size] = it->ng();
                path_size++;
            });

            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { $break; };

            $if(stop_direct) {
                auto it_next = pipeline().geometry()->intersect(ray);
                $if(!it_next->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    }
                };
                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it_next->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it_next, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    };
                }
                flag=true;
                end_point = it->p();
                $break;
            };
            $if(depth + 1u >= rr_depth) {
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };

        $if(flag==true){
            ray = camera_ray;
            $for(i,path_size)
            {
                Float3 point_cur = points[i], point_nxt, point_pre;
                Float3 normal_cur = normals[i];

                $if(i > 0) {
                    point_pre = points[i - 1];
                } $else {
                    point_pre = start_point;
                };

                $if(i<path_size-1){
                    point_nxt = points[i];
                } $else{
                    point_nxt = end_point;
                };
                PolymorphicCall<Surface::Closure> call;

                auto wo = normalize(point_pre-point_cur);
                auto it = pipeline().geometry()->intersect(ray);

                ray = it->spawn_ray(normalize(point_nxt-point_cur));
                pipeline().surfaces().dispatch(it->shape().surface_tag(), [&](auto surface) noexcept {
                    surface->closure(call, *it, swl, wo, 1.f, time);
                });
                Float3 grad_p_cur, grad_p_nxt, grad_p_pre;
                call.execute([&](auto closure) noexcept {
                    Float3 wi = normalize(point_nxt-point_cur);
                    Float3 wo = normalize(point_pre-point_cur);
                    auto eval = closure->evaluate(wo, wi);
                    auto evalf = make_float3(eval.f[0u],eval.f[1u],eval.f[2u]);
                    auto devalf = ite(evalf == 0.f, make_float3(0.f), make_float3(radiance[0u] / evalf[0u],radiance[1u] / evalf[1u],radiance[2u] / evalf[2u]));
                    SampledSpectrum a{3u},b{3u};
                    a[0u] = grad_rgb[0u];
                    a[1u] = grad_rgb[1u];
                    a[2u] = grad_rgb[2u];
                    b[0u] = devalf[0u];
                    b[1u] = devalf[1u];
                    b[2u] = devalf[2u];
                    closure->backward(wo, wi, a*b);
                    $autodiff{
                        requires_grad(point_cur, point_nxt, point_pre);
                        wi = normalize(point_nxt-point_cur);
                        wo = normalize(point_pre-point_cur);
                        auto eval = closure->evaluate(wo, wi);
                        evalf = make_float3(eval.f[0u], eval.f[1u], eval.f[2u]);
                        backward(grad_rgb*devalf*evalf);
                        grad_p_cur = grad(point_cur);
                        grad_p_nxt = grad(point_nxt);
                        grad_p_pre = grad(point_pre);
                    };
                });

                $if(i>0){
                    grad_barys[i-1]+=get_bary_grad(grad_p_pre, inst_ids[i-1], triangle_ids[i-1]);
                };
                grad_barys[i]+=get_bary_grad(grad_p_cur, inst_ids[i], triangle_ids[i]);
                $if(i<path_size-1){
                    grad_barys[i+1]+=get_bary_grad(grad_p_nxt, inst_ids[i], triangle_ids[i]);
                };

            };
        };
        EPSM_path(path_size-1, grad_barys, start_point, end_point, triangle_ids, inst_ids, bary_coords, etas, points, normals);
        return spectrum->srgb(swl, Li);
    }

    [[nodisgard]] Float2 get_bary_grad(Float3 grad_in, UInt inst_id, UInt triangle_id){
        auto instance = pipeline().geometry()->instance(inst_id);
        auto triangle = pipeline().geometry()->triangle(instance, triangle_id);
        auto v_buffer = instance.vertex_buffer_id();

        auto v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
        auto v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
        auto v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);

        Float2 grad_bary;
        grad_bary[0] = dot(v0->position() - v2->position(), grad_in);
        grad_bary[1] = dot(v1->position() - v2->position(), grad_in);
        return grad_bary;
    };
    
    [[nodiscard]] void photon_tracing_bp(const Camera::Instance *camera, Expr<uint> frame_index,
                        Expr<uint2> sampler_id, Expr<float> time, Expr<uint> photon_id_1d, BufferFloat &grad_in) {

        sampler()->start(sampler_id, frame_index);
        // generate uniform samples
        auto u_light_selection = sampler()->generate_1d();
        auto u_light_surface = sampler()->generate_2d();
        auto u_direction = sampler()->generate_2d();
        auto spectrum = pipeline().spectrum();
        auto swl = spectrum->sample(spectrum->node()->is_fixed() ? 0.f : sampler()->generate_1d());
        auto light_sample = light_sampler()->sample_le(
            u_light_selection, u_light_surface, u_direction, swl, time);
        //cos term canceled out in pdf
        SampledSpectrum beta = light_sample.eval.L / light_sample.eval.pdf;
        
        auto ray = light_sample.shadow_ray;
        auto pdf_bsdf = def(1e16f);

        //logger->add_start_light(photon_id_1d, light_sample);
        UInt path_size = 0u; 

        ArrayUInt<4> triangle_ids, inst_ids;
        ArrayFloat3<4> bary_coords, points, normals;
        ArrayFloat<4> etas; 
        ArrayFloat2<4> grad_barys;
        ArrayFloat3<4> grad_betas;
    
        auto resolution = camera->film()->node()->resolution();
        auto max_depth = min(node<MegakernelPhotonMappingDiff>()->max_depth(), 4u);
        auto tot_neighbors = 0u;

        $for(depth, max_depth) {

            auto wi = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);
            grad_barys[path_size] = make_float2(0.f);
            grad_betas[path_size] = make_float3(0.f);

            $if(!it->valid()) {
                $break;
            };

            $if(!it->shape().has_surface()) { 
                $break; 
            };
            // generate uniform samples
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<MegakernelPhotonMappingDiff>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            $if(depth > 0) {// add diffuse constraint?
                auto grid = viewpoints->point_to_grid(it->p());
                Float3 grad_beta = make_float3(0.f);
                Float2 grad_bary = make_float2(0.f);
                auto count_neighbors = 0u;
                $for(x, grid.x - 1, grid.x + 2) {
                    $for(y, grid.y - 1, grid.y + 2) {
                        $for(z, grid.z - 1, grid.z + 2) {
                            Int3 check_grid{x, y, z};
                            auto viewpoint_index = viewpoints->grid_head(viewpoints->grid_to_index(check_grid));
                            $while(viewpoint_index != ~0u) {
                                auto position = viewpoints->position(viewpoint_index);
                                auto pixel_id = viewpoints->pixel_id(viewpoint_index);
                                
                                auto dis = distance(position, it->p());
                                auto rad = indirect->radius(pixel_id);
                                $if(dis <= rad) {
                                    auto viewpoint_beta = viewpoints->beta(viewpoint_index);
                                    auto viewpoint_wo = viewpoints->wo(viewpoint_index);
                                    auto bary = it->bary_coord();
                                    auto instance = pipeline().geometry()->instance(it->instance_id());
                                    auto triangle = pipeline().geometry()->triangle(instance, it->triangle_id());
                                    auto v_buffer = instance.vertex_buffer_id();
                                    auto v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
                                    auto v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
                                    auto v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);
                                    auto point_0 = v0->position();
                                    auto point_1 = v1->position();
                                    auto point_2 = v2->position();
                                    auto surface_tag = viewpoints->surface_tag(viewpoint_index);
                                    SampledSpectrum eval_viewpoint(3u);
                                    PolymorphicCall<Surface::Closure> call;
                                    pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                                        surface->closure(call, *it, swl, viewpoint_wo, 1.f, time);
                                    });
                                    call.execute([&](const Surface::Closure *closure) noexcept {
                                        eval_viewpoint = closure->evaluate(viewpoint_wo, wi).f; 
                                    });
                                    Float2 grad_b{0.0f, 0.0f};
                                    Float grad_pixel_0, grad_pixel_1, grad_pixel_2;
                                    Float rel_dis_diff, grad_dis;
                                    Float weight;
                                    Float3 Phi, Phi_real;
                                    $autodiff {
                                        Float3 beta_diff = make_float3(beta[0u], beta[1u], beta[2u]);
                                        requires_grad(bary, beta_diff);
                                        Float3 photon_pos = point_0 * bary[0] + point_1 * bary[1] + point_2 * (1 - bary[0] - bary[1]);
                                        rel_dis_diff = distance(position, photon_pos) / rad;
                                        requires_grad(rel_dis_diff);
                                        auto rel3 = rel_dis_diff*rel_dis_diff*rel_dis_diff;
                                        weight = 3.5f*(1- 6*rel3*rel_dis_diff*rel_dis_diff + 15*rel3*rel_dis_diff - 10*rel3);
                                        auto wi_local = it->shading().world_to_local(wi);
                                        Phi = spectrum->srgb(swl, viewpoint_beta * eval_viewpoint / abs_cos_theta(wi_local));
                                        Phi_real = spectrum->srgb(swl, beta * viewpoint_beta * eval_viewpoint / abs_cos_theta(wi_local));
                                        auto Phi_beta = Phi * beta_diff * weight / (indirect->_photon_per_iter*1.0f);
                                        grad_pixel_0 = grad_in->read(pixel_id * _grad_dimension + 0);
                                        grad_pixel_1 = grad_in->read(pixel_id * _grad_dimension + 1);
                                        grad_pixel_2 = grad_in->read(pixel_id * _grad_dimension + 2);
                                        auto dldPhi = (Phi_beta[0u]*grad_pixel_0 + Phi_beta[1u]*grad_pixel_1 + Phi_beta[2u]*grad_pixel_2);
                                        backward(dldPhi);
                                        grad_bary += grad(bary).xy();
                                        grad_beta += grad(beta_diff);
                                        grad_dis = grad(rel_dis_diff);
                                    };
                                    count_neighbors+=1;

                                    $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 2) % 2 == 1) {
                                        $if(photon_id_1d<node<MegakernelPhotonMappingDiff>()->debug_photon())
                                        {
                                            $if(etas[0]==1.8f)
                                            {
                                                UInt2 pixel_id_2d = make_uint2(pixel_id % resolution.x, pixel_id / resolution.x);
                                                camera->film()->accumulate(pixel_id_2d, make_float3(grad_dis,0.0f,0.0f));
                                            };
                                        };
                                    };

                                    $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 16) % 2 == 1) {
                                        $if(photon_id_1d<node<MegakernelPhotonMappingDiff>()->debug_photon())
                                        {
                                            indirect->add_phi(pixel_id, Phi_real*weight);
                                            indirect->add_cur_n(pixel_id, 1u);
                                        };
                                    };
                                };
                                viewpoint_index = viewpoints->nxt(viewpoint_index);
                            };
                        };
                    };
                };
                $if(count_neighbors>0){
                    tot_neighbors+=count_neighbors;
                    grad_betas[path_size] += make_float3(grad_beta[0]/count_neighbors, grad_beta[1]/count_neighbors, grad_beta[2]/count_neighbors);
                    grad_barys[path_size] += make_float2(grad_bary[0]/count_neighbors, grad_bary[1]/count_neighbors);
                };
            };
            auto surface_tag = it->shape().surface_tag();
            auto eta_scale = def(1.f);
            PolymorphicCall<Surface::Closure> call;
            pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                surface->closure(call, *it, swl, wi, 1.f, time);
            });
            call.execute([&](auto closure) noexcept {
                // sample material
                auto surface_sample = closure->sample(wi, u_lobe, u_bsdf, TransportMode::IMPORTANCE);
                ray = it->spawn_ray(surface_sample.wi);
                pdf_bsdf = surface_sample.eval.pdf;
                auto w = ite(surface_sample.eval.pdf > 0.f, 1.f / surface_sample.eval.pdf, 0.f);
                auto bnew = beta * w * surface_sample.eval.f;
                // apply eta scale
                auto eta = closure->eta().value_or(1.f);
                auto roughness = closure->roughness();
                $switch(surface_sample.event) {
                    $case(Surface::event_enter) { eta_scale = sqr(eta);  etas[path_size] = eta;};
                    $case(Surface::event_exit) { eta_scale = sqr(1.f / eta);  etas[path_size] = 1.f / eta;};
                };
                eta_scale *= ite(beta.max() < bnew.max(), 1.f, bnew.max() / beta.max());
                beta = bnew;
            });
            beta = zero_if_any_nan(beta);
            $if(beta.all([](auto b) noexcept { return b <= 0.f; })) { 
                $break; 
            };
            auto rr_threshold = node<MegakernelPhotonMappingDiff>()->rr_threshold();
            auto q = max(eta_scale, .05f);
            $if(depth + 1u >= rr_depth) {
                $if(q < rr_threshold & u_rr >= q) {
                    $break; 
                };
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };

            triangle_ids[path_size] = it->triangle_id();
            inst_ids[path_size] = it->instance_id();
            bary_coords[path_size] = it->bary_coord();
            points[path_size] = it->p();
            normals[path_size] = it->ng();
            path_size = path_size+1u;
        };
        //@Todo: log surface event
        $if(photon_id_1d<node<MegakernelPhotonMappingDiff>()->debug_photon())
        {
            $if(path_size==2) {
                $if(etas[0]==1.8f)
                {
                    $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 1) % 2 == 1) {
                        device_log("path size is {}",path_size);
                        device_log("photon_id is {}",photon_id_1d);
                        device_log("light sample is {}",light_sample.shadow_ray->origin());
                        $for(i,path_size)
                        {    
                            device_log("{} point {} normal {} eta {} inst {}, bary {} grad_bary {}",i, points[i], normals[i], etas[i], inst_ids[i], bary_coords[i], grad_barys[i]);
                        };
                    };
                    EPSM_photon(path_size, points, normals, inst_ids, triangle_ids, bary_coords, etas, light_sample.shadow_ray->origin(), grad_barys, photon_id_1d);
                };
            };
        };
    }
    
    struct GradStruct {
        Float3 grad_p_pre;
        Float3 grad_p_cur;
        Float3 grad_p_nxt;
        Float3 grad_n_cur;
        Float   grad_eta;
    };

    GradStruct half_vec_constraint(Float3 point_pre, Float3 point_cur, Float3 point_nxt, Float3 normal_cur, Float eta, size_t j) {
        Float3 grad_p_pre{0.f, 0.f, 0.f}, grad_p_cur{0.f, 0.f, 0.f}, grad_p_nxt{0.f, 0.f, 0.f}, grad_n_cur{0.f, 0.f, 0.f};
        Float grad_eta{0.f};
        $autodiff{
            requires_grad(point_pre, point_cur, point_nxt, normal_cur, eta);
            auto wi = normalize(point_cur-point_nxt);
            auto wo = normalize(point_nxt-point_cur);
            auto f = Frame::make(normal_cur);
            auto wi_local = f.world_to_local(wi);
            auto wo_local = f.world_to_local(wo);
            auto res = normalize(wi_local+wo_local*eta);
            backward(res[j]);
            grad_p_pre = grad(point_pre);
            grad_p_cur = grad(point_cur);
            grad_p_nxt = grad(point_nxt);
            grad_n_cur = grad(normal_cur);
            grad_eta = grad(eta);
        };
        return {grad_p_pre, grad_p_cur, grad_p_nxt, grad_n_cur, grad_eta};
    }

    auto locate(UInt i, UInt j) {
        return (i*max_size*4u+j);
    };
    
    auto locate_adj(UInt i, UInt j) {
        return (i*max_size*4+max_size*2+j);
    };

    void inverse_matrix(ArrayFloat<8*8*2> mat_bary, UInt path_size){
        auto n = path_size*2;
        $for (i,n) {
            mat_bary[locate_adj(i,i)] = 1;
        };
        // ArrayFloat<8 * 8 * 2> mat_bary_copy{mat_bary};
        // $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 1) % 2 == 1)
        // {
        //     $for (i,8) {
        //         device_log("mat_bary {} {} {} {} {} {} {} {}",mat_bary[locate(i,0)],mat_bary[locate(i,1)],mat_bary[locate(i,2)],mat_bary[locate(i,3)],mat_bary[locate(i,4)],mat_bary[locate(i,5)],mat_bary[locate(i,6)],mat_bary[locate(i,7)]);
        //     };
        // };
        $for (i, n) {
            $if (mat_bary[locate(i,i)]==0) {
                auto p=n;
                $for (j,i + 1, n) {
                    $if (mat_bary[locate(j,i)] != 0) {
                        p=j;
                        $break;
                    };
                };
                $for(k,n)
                {
                    auto t = mat_bary[locate(i, k)];
                    mat_bary[locate(i,k)] = mat_bary[locate(p,k)];
                    mat_bary[locate(p,k)] = t;
                    
                    t = mat_bary[locate_adj(i,k)];
                    mat_bary[locate_adj(i,k)] = mat_bary[locate_adj(p,k)];
                    mat_bary[locate_adj(p,k)] = t;
                };
            };
            $for(j, n) {
                $if(j!=i)
                {
                    auto factor = mat_bary[locate(j,i)]/mat_bary[locate(i,i)];
                    $for (k,i,n) {
                        mat_bary[locate(j,k)] = mat_bary[locate(j,k)] - factor * mat_bary[locate(i,k)];
                    };
                    $for (k,n) {
                        mat_bary[locate_adj(j,k)] = mat_bary[locate_adj(j,k)] - factor * mat_bary[locate_adj(i,k)];
                    };
                };
            };
        };
        $for(i, n) {
            auto f = mat_bary[locate(i,i)];
            $for(j,i,n) {
                mat_bary[locate(i,j)] /= f;
            };
            $for(j, n) {
                mat_bary[locate_adj(i,j)] /= f;
            };
        };
        // $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 1) % 2 == 1)
        // {
        //     $for (i,8){
        //         $for(j,8)
        //         {
        //             mat_bary_copy[locate_adj(i,j)] = 0;
        //             $for (k,8) {
        //                 mat_bary_copy[locate_adj(i,j)]+=mat_bary_copy[locate(i,k)]*mat_bary[locate_adj(k,j)];
        //             };
        //         };
        //         device_log("should be id {} {} {} {} {} {} {} {}",mat_bary_copy[locate_adj(i,0)],mat_bary_copy[locate_adj(i,1)],mat_bary_copy[locate_adj(i,2)],mat_bary_copy[locate_adj(i,3)],mat_bary_copy[locate_adj(i,4)],mat_bary_copy[locate_adj(i,5)],mat_bary_copy[locate_adj(i,6)],mat_bary_copy[locate_adj(i,7)]);
        //     };
        //     $for (i,8) {
        //         device_log("mat_bary {} {} {} {} {} {} {} {}",mat_bary[locate(i,0)],mat_bary[locate(i,1)],mat_bary[locate(i,2)],mat_bary[locate(i,3)],mat_bary[locate(i,4)],mat_bary[locate(i,5)],mat_bary[locate(i,6)],mat_bary[locate(i,7)]);
        //     };
        // };
    }
    
    void compute_and_scatter_grad(ArrayFloat<8 * 8 * 2> &mat_bary, ArrayFloat3<8 * 5 + 2 * 2> &mat_param, ArrayFloat2<4> &grad_bary, UInt path_size, ArrayUInt<4> &inst_ids, ArrayUInt<4> &triangle_ids, ArrayFloat3<4> &bary_coords) {
        ArrayFloat<8> tmp;
        auto n = path_size*2;
        $for(i, n){
            tmp[i] = 0.0f;
            $for(j, n/2){
                tmp[i] -= grad_bary[j][0] * mat_bary[locate_adj(j * 2, i)] + grad_bary[j][1] * mat_bary[locate_adj(j * 2 + 1, i)];
            };
        }; 
        $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 1) % 2 == 1)
        {
            $for (i,8) {
                device_log("mat_param {} {} {} {}",mat_param[i*4+0],mat_param[i*4+1],mat_param[i*4+2],mat_param[i*4+3]);
            };
            $for(j, n/2){
                device_log("grad_bary {} {} ",grad_bary[j][0], grad_bary[j][1]);
            };
        };
        //Todo add eta support
        $for(i, n/2) {
            Float3 grad_vertex = make_float3(0.0f), grad_normal = make_float3(0.0f);
            grad_vertex = tmp[i*2] * mat_param[((i * 2) << 2) + 2] + tmp[i*2+1] * mat_param[((i * 2 + 1) << 2) + 2];
            $if(i<n/2-1){
                grad_vertex += tmp[(i+1)*2]*mat_param[(((i+1)*2)<<2)+1] + tmp[(i+1)*2+1]*mat_param[(((i+1)*2+1)<<2)+1];
                grad_normal  = tmp[(i+1)*2]*mat_param[(((i+1)*2)<<2)+3] + tmp[(i+1)*2+1]*mat_param[(((i+1)*2+1)<<2)+3];
            };
            $if(i<n/2-2){
                grad_vertex += tmp[(i+2)*2]*mat_param[(((i+2)*2)<<2)] + tmp[(i+2)*2+1]*mat_param[(((i+2)*2+1)<<2)];
            };
            $if(inst_ids[i]==0)
            {
                $if((node<MegakernelPhotonMappingDiff>()->debug_mode() / 1) % 2 == 1)
                {
                    device_log("grad_vertex {} grad_normal {} bary {}",grad_vertex, grad_normal, bary_coords[i]);
                };
                pipeline().differentiation()->add_geom_gradients(grad_vertex, grad_normal, bary_coords[i], inst_ids[i], triangle_ids[i]);
            };                
        };
    }
    
    void EPSM_path(UInt path_size, ArrayFloat2<4> grad_bary, Float3 start_point, Float3 end_point, ArrayUInt<4> &inst_ids, ArrayUInt<4> &triangle_ids, ArrayFloat3<4> &bary_coords, ArrayFloat<4> &etas, ArrayFloat3<4> &points, ArrayFloat3<4> &normals) {
        ArrayFloat<8*8*2> mat_bary;
        ArrayFloat3<8*5+2*2> mat_param;
        Float3 point_pre, point_cur, point_nxt, normal_cur;

        $for(id, path_size){

            Float3 bary_cur = bary_coords[id];
            Float3 bary_nxt = bary_coords[id+1];

            point_cur = points[id];
            normal_cur = normals[id];

            $if (id>0){
                point_pre = points[id-1];
            } $else{
                point_pre = start_point;
            };

            $if(id<path_size-1){
                point_nxt = points[id+1];
            } $else{
                point_nxt = end_point;
            };
            
            for(uint j=0;j<2;j++){
                auto [grad_p_pre, grad_p_cur, grad_p_nxt, grad_n_cur, grad_eta] = half_vec_constraint(point_pre, point_cur, point_nxt, normal_cur, etas[id], j);
                auto grad_b_cur = get_bary_grad(grad_p_cur+grad_n_cur, inst_ids[id], triangle_ids[id]);
                mat_bary[locate(id*2+j,id*2+0)] = grad_b_cur[0];
                mat_bary[locate(id*2+j,id*2+1)] = grad_b_cur[1];

                mat_param[(id*2+j)*param_size_per_vert+1] = grad_p_cur;
                mat_param[(id*2+j)*param_size_per_vert+3] = grad_n_cur;
                mat_param[(id*2+j)*param_size_per_vert+4] = make_float3(grad_eta);

                $if (id>0){
                    auto grad_b_pre = get_bary_grad(grad_p_pre, inst_ids[id-1], triangle_ids[id-1]);
                    mat_bary[locate(id*2+j,id*2-2)] = grad_b_pre[0];
                    mat_bary[locate(id*2+j,id*2-1)] = grad_b_pre[1];
                    mat_param[(id*2+j)*param_size_per_vert+0] = grad_p_pre;
                } $else{
                    mat_param[max_size*param_size_per_vert+j] = grad_p_pre;
                };

                $if (id<path_size-1){
                    auto grad_b_nxt = get_bary_grad(grad_p_nxt, inst_ids[id+1], triangle_ids[id+1]);
                    mat_bary[locate(id*2+j,id*2+2)] = grad_b_nxt[0];
                    mat_bary[locate(id*2+j,id*2+3)] = grad_b_nxt[1];
                    mat_param[(id*2+j)*param_size_per_vert+2] = grad_p_nxt;
                } $else{
                    mat_param[max_size*param_size_per_vert*2+j] = grad_p_nxt;
                };
            }
        };
        inverse_matrix(mat_bary, path_size);
        compute_and_scatter_grad(mat_bary, mat_param, grad_bary, path_size, inst_ids, triangle_ids, bary_coords);
    }
    
    void EPSM_photon(UInt path_size, ArrayFloat3<4> &points, ArrayFloat3<4> &normals, ArrayUInt<4> &inst_ids, ArrayUInt<4> &triangle_ids, ArrayFloat3<4> &bary_coords, ArrayFloat<4> &etas, Float3 start_point, ArrayFloat2<4> grad_bary, UInt photon_id_1d){
    {
        ArrayFloat<8*8*2> mat_bary;
        ArrayFloat3<8*5+2*2> mat_param;
        Float3 point_pre, point_cur, point_nxt, normal_cur;
        $for(id, path_size-1){

            Float3 bary_cur = bary_coords[id];
            Float3 bary_nxt = bary_coords[id+1];

            point_cur = points[id];
            normal_cur = normals[id];

            $if (id>0){
                point_pre = points[id-1];
            } $else{
                point_pre = start_point;
            };

            point_nxt = points[id+1];
            
            $if(id==0) {
                Float3 bary_cur = bary_coords[0];  
                auto instance = pipeline().geometry()->instance(inst_ids[id]);
                auto triangle = pipeline().geometry()->triangle(instance, triangle_ids[id]);
                auto v_buffer = instance.vertex_buffer_id();

                auto v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
                auto v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
                auto v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);

                auto point_cur_0 = v0->position();
                auto point_cur_1 = v1->position();
                auto point_cur_2 = v2->position();
                $autodiff{
                    requires_grad(bary_cur);
                    Float3 point_cur = point_cur_0*bary_cur[0]+point_cur_1*bary_cur[1]+point_cur_2*(1-bary_cur[0]-bary_cur[1]);
                    requires_grad(point_cur);
                    auto wi_diff = normalize(point_cur-point_pre);
                    backward(wi_diff[0]);
                    mat_param[2] = grad(point_cur);
                    auto grad_b = grad(bary_cur);
                    mat_bary[locate(0,0)] = grad_b[0];
                    mat_bary[locate(0,1)] = grad_b[1];
                };
                $autodiff {
                    requires_grad(bary_cur);
                    Float3 point_cur = point_cur_0 * bary_cur[0] + point_cur_1 * bary_cur[1] + point_cur_2 * (1 - bary_cur[0] - bary_cur[1]);
                    requires_grad(point_cur);
                    auto wi_diff = normalize(point_cur - point_pre);
                    backward(wi_diff[1]);
                    mat_param[6] = grad(point_cur);
                    auto grad_b = grad(bary_cur);
                    mat_bary[locate(1, 0)] = grad_b[0];
                    mat_bary[locate(1, 1)] = grad_b[1];
                };
            };
            uint offset = 2u;
            for(uint j=0;j<2;j++){
                auto [grad_p_pre, grad_p_cur, grad_p_nxt, grad_n_cur, grad_eta] = half_vec_constraint(point_pre, point_cur, point_nxt, normal_cur, etas[id], j);
                auto grad_b_cur = get_bary_grad(grad_p_cur+grad_n_cur, inst_ids[id], triangle_ids[id]);
                mat_bary[locate(offset+id*2+j,id*2+0)] = grad_b_cur[0];
                mat_bary[locate(offset+id*2+j,id*2+1)] = grad_b_cur[1];

                mat_param[(offset+id*2+j)*param_size_per_vert+1] = grad_p_cur;
                mat_param[(offset+id*2+j)*param_size_per_vert+3] = grad_n_cur;
                mat_param[(offset + id * 2 + j) * param_size_per_vert + 4] = make_float3(grad_eta);

                $if (id>0){
                    auto grad_b_pre = get_bary_grad(grad_p_pre, inst_ids[id-1], triangle_ids[id-1]);
                    mat_bary[locate(offset+id*2+j,id*2-2)] = grad_b_pre[0];
                    mat_bary[locate(offset+id*2+j,id*2-1)] = grad_b_pre[1];
                    mat_param[(offset+id*2+j)*param_size_per_vert+0] = grad_p_pre;
                };

                $if (id<path_size-1){
                    auto grad_b_nxt = get_bary_grad(grad_p_nxt, inst_ids[id+1], triangle_ids[id+1]);
                    mat_bary[locate(offset+id*2+j,id*2+2)] = grad_b_nxt[0];
                    mat_bary[locate(offset+id*2+j,id*2+3)] = grad_b_nxt[1];
                    mat_param[(offset+id*2+j)*param_size_per_vert+2] = grad_p_nxt;
                };
            }
        };
        inverse_matrix(mat_bary, path_size);
        compute_and_scatter_grad(mat_bary, mat_param, grad_bary, path_size, inst_ids, triangle_ids, bary_coords);
    }
};

};

luisa::unique_ptr<Integrator::Instance> MegakernelPhotonMappingDiff::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelPhotonMappingDiffInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPhotonMappingDiff)
