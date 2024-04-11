//
// Created by Jiankai on 2024/3/4.
//

#include <util/sampling.h>
#include <util/medium_tracker.h>
#include <util/progress_bar.h>
#include <base/pipeline.h>
#include <base/integrator.h>

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
class MegakernelPhotonMappingDiff final : public DifferentiableIntegrator {

private:
    uint _max_depth;
    uint _rr_depth;
    float _rr_threshold;
    uint _photon_per_iter;
    float _initial_radius;
    bool _separate_direct;
    bool _shared_radius;

public:
    MegakernelPhotonMappingDiff(Scene *scene, const SceneNodeDesc *desc) noexcept
        : DifferentiableIntegrator{scene, desc},
          _max_depth{std::max(desc->property_uint_or_default("depth", 10u), 1u)},
          _rr_depth{std::max(desc->property_uint_or_default("rr_depth", 2u), 0u)},
          _rr_threshold{std::max(desc->property_float_or_default("rr_threshold", 0.95f), 0.05f)},
          _initial_radius{std::max(desc->property_float_or_default("initial_radius", -200.f), -10000.f)},//<0 for world_size/-radius (-grid count)
          _photon_per_iter{std::max(desc->property_uint_or_default("photon_per_iter", 200000u), 10u)},
          _separate_direct{true},                                                  //when false, use photon mapping for all flux and gathering at first intersection. Just for debug
          _shared_radius{desc->property_bool_or_default("shared_radius", true)} {};//whether or not use the shared radius trick in SPPM paper. True is better in performance.
    [[nodiscard]] auto max_depth() const noexcept { return _max_depth; }
    [[nodiscard]] auto photon_per_iter() const noexcept { return _photon_per_iter; }
    [[nodiscard]] auto rr_depth() const noexcept { return _rr_depth; }
    [[nodiscard]] auto initial_radius() const noexcept { return _initial_radius; }
    [[nodiscard]] auto rr_threshold() const noexcept { return _rr_threshold; }
    [[nodiscard]] bool is_differentiable() const noexcept override { return true; }
    [[nodiscard]] auto separate_direct() const noexcept { return _separate_direct; }
    [[nodiscard]] auto shared_radius() const noexcept { return _shared_radius; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MegakernelPhotonMappingDiffInstance final : public ProgressiveIntegrator::Instance {

public:
    using ProgressiveIntegrator::Instance::Instance;
    //A hash table for storing photons in grid
    //Some problem:can only initialize for super large photon cache(max_depth*photon_per_iter)
    //The fetchmax functions have wrong implementation in Luisa compute, so related feature are wrong now
    //(Including uint grid index, and inital_radius<0)
    class ViewPointMap {
    private:
        Buffer<uint> _grid_head;
        Buffer<float> _beta;
        Buffer<float3> _wo;
        Buffer<float3> _position;
        Buffer<uint2> _pixel_id;
        Buffer<uint> _mat_type;
        Buffer<uint> _nxt;
        uint _size;//current photon count
        const Spectrum::Instance *_spectrum;
        Buffer<float> _grid_min;//atomic float3
        Buffer<float> _grid_max;//atomic float3
        Buffer<float> _grid_len;//the length of a single grid (float1)
        Buffer<float> _swl_lambda;
        Buffer<float> _swl_pdf;
        Buffer<uint> _tot;

    public:
        ViewPointMap(uint viewpoint_count, const Spectrum::Instance *spectrum) {
            auto &&device = spectrum->pipeline().device();
            _grid_head = device.create_buffer<uint>(viewpoint_count);
            _beta = device.create_buffer<float>(viewpoint_count * spectrum->node()->dimension());
            _wo = device.create_buffer<float3>(viewpoint_count);
            _position = device.create_buffer<float3>(viewpoint_count);
            _pixel_id = device.create_buffer<uint2>(viewpoint_count);
            _mat_type = device.create_buffer<uint>(viewpoint_count);
            _nxt = device.create_buffer<uint>(viewpoint_count);
            _tot = device.create_buffer<uint>(1u);
            _grid_len = device.create_buffer<float>(1u);
            _grid_min = device.create_buffer<float>(3u);
            _grid_max = device.create_buffer<float>(3u);
            _size = viewpoint_count;
            _spectrum = spectrum;
            if (!_spectrum->node()->is_fixed()) {
                _swl_lambda = device.create_buffer<float>(viewpoint_count * spectrum->node()->dimension());
                _swl_pdf = device.create_buffer<float>(viewpoint_count * spectrum->node()->dimension());
            }
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
            auto dimension = _spectrum->node()->dimension();
            SampledSpectrum s{dimension};
            for (auto i = 0u; i < dimension; ++i)
                s[i] = _beta->read(index * dimension + i);
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
        auto swl(Expr<uint> index) const noexcept {
            auto dimension = _spectrum->node()->dimension();
            SampledWavelengths swl(dimension);
            for (auto i = 0u; i < dimension; ++i) {
                swl.set_lambda(i, _swl_lambda->read(index * dimension + i));
                swl.set_pdf(i, _swl_pdf->read(index * dimension + i));
            }
            return swl;
        }
        void push(Expr<float3> position, SampledWavelengths swl, SampledSpectrum power, Expr<float3> wi, Expr<uint2> pixel_id, Surface::Closure* closure) {
            auto index = _tot->atomic(0u).fetch_add(1u);
            auto dimension = _spectrum->node()->dimension();
            if (!_spectrum->node()->is_fixed()) {
                for (auto i = 0u; i < dimension; ++i) {
                    _swl_lambda->write(index * dimension + i, swl.lambda(i));
                    _swl_pdf->write(index * dimension + i, swl.pdf(i));
                }
            }

            _wi->write(index, wi);
            _position->write(index, position);
            _pixel_id->write(index, pixel_id);
            _mat_type->write(index, 0);
            for (auto i = 0u; i < dimension; ++i)
                _beta->write(index * dimension + i, power[i]);
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
        Buffer<float> _radius;
        Buffer<uint> _cur_n;
        Buffer<uint> _n_photon;
        Buffer<float> _phi;
        Buffer<float> _tau;
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
                _n_photon = device.create_buffer<uint>(1);
            } else {
                _radius = device.create_buffer<float>(resolution.x * resolution.y);
                _cur_n = device.create_buffer<uint>(resolution.x * resolution.y);
                _n_photon = device.create_buffer<uint>(resolution.x * resolution.y);
            }
            _phi = device.create_buffer<float>(resolution.x * resolution.y * dimension);
            _tau = device.create_buffer<float>(resolution.x * resolution.y * dimension);
            _photon_per_iter = photon_per_iter;
        }
        void write_radius(Expr<uint2> pixel_id, Expr<float> value) noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                _radius->write(pixel_id.y * resolution.x + pixel_id.x, value);
            } else {
                _radius->write(0u, value);
            }
        }
        void write_cur_n(Expr<uint2> pixel_id, Expr<uint> value) noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                _cur_n->write(pixel_id.y * resolution.x + pixel_id.x, value);
            } else {
                _cur_n->write(0u, value);
            }
        }
        void write_n_photon(Expr<uint2> pixel_id, Expr<uint> value) noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                _n_photon->write(pixel_id.y * resolution.x + pixel_id.x, value);
            } else {
                _n_photon->write(0u, value);
            }
        }
        void reset_phi(Expr<uint2> pixel_id) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            for (auto i = 0u; i < dimension; ++i)
                _phi->write(offset * dimension + i, 0.f);
        }
        void reset_tau(Expr<uint2> pixel_id) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            for (auto i = 0u; i < dimension; ++i)
                _tau->write(offset * dimension + i, 0.f);
        }
        auto radius(Expr<uint2> pixel_id) const noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                return _radius->read(pixel_id.y * resolution.x + pixel_id.x);
            } else {
                return _radius->read(0u);
            }
        }
        //tau=(tau+clamp(phi))*value, see pixel_info_update for useage
        void update_tau(Expr<uint2> pixel_id, Expr<float> value) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            auto thershold = _clamp;
            for (auto i = 0u; i < dimension; ++i) {
                auto old_tau = _tau->read(offset * dimension + i);
                auto phi = _phi->read(offset * dimension + i);
                phi = max(-thershold, min(phi, thershold));//-thershold for wavelength sampling
                _tau->write(offset * dimension + i, (old_tau + phi) * value);
            }
        }

        auto n_photon(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            if (!_shared_radius) {
                return _n_photon->read(pixel_id.y * resolution.x + pixel_id.x);
            } else {
                return _n_photon->read(0u);
            }
        }
        auto cur_n(Expr<uint2> pixel_id) const noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                return _cur_n->read(pixel_id.y * resolution.x + pixel_id.x);
            } else {
                return _cur_n->read(0u);
            }
        }
        auto phi(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            Float3 ret;
            for (auto i = 0u; i < dimension; ++i)
                ret[i] = _phi->read(offset * dimension + i);
            return ret;
        }
        auto tau(Expr<uint2> pixel_id) const noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            Float3 ret;
            for (auto i = 0u; i < dimension; ++i)
                ret[i] = _tau->read(offset * dimension + i);
            return ret;
        }
        void add_cur_n(Expr<uint2> pixel_id, Expr<uint> value) noexcept {
            if (!_shared_radius) {
                auto resolution = _film->node()->resolution();
                auto offset = pixel_id.y * resolution.x + pixel_id.x;
                _cur_n->atomic(offset).fetch_add(value);
            } else {
                _cur_n->atomic(0u).fetch_add(value);
            }
        }
        void add_phi(Expr<uint2> pixel_id, Expr<float3> phi) noexcept {
            auto resolution = _film->node()->resolution();
            auto offset = pixel_id.y * resolution.x + pixel_id.x;
            auto dimension = 3u;
            for (auto i = 0u; i < dimension; ++i)
                _phi->atomic(offset * dimension + i).fetch_add(phi[i]);
        }
        void pixel_info_update(Expr<uint2> pixel_id) {
            $if(cur_n(pixel_id) > 0) {
                Float gamma = 2.0f / 3.0f;
                UInt n_new = n_photon(pixel_id) + cur_n(pixel_id);
                Float r_new = radius(pixel_id) * sqrt(n_new * gamma / (n_photon(pixel_id) * gamma + cur_n(pixel_id)));
                //indirect.write_tau(pixel_id, (indirect.tau(pixel_id) + indirect.phi(pixel_id)) * (r_new * r_new) / (indirect.radius(pixel_id) * indirect.radius(pixel_id)));
                update_tau(pixel_id, r_new * r_new / (radius(pixel_id) * radius(pixel_id)));
                if (!_shared_radius) {
                    write_n_photon(pixel_id, n_new);
                    write_cur_n(pixel_id, 0u);
                    write_radius(pixel_id, r_new);
                }
                reset_phi(pixel_id);
            };
        }
        void shared_update() {
            auto pixel_id = make_uint2(0, 0);
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
    
    class PhotonMappingLogger{
    public:

        Buffer<UInt> photon_inst_ids, photon_triangle_ids, photon_scatter_events, photon_sizes;
        Buffer<UInt> path_inst_ids, path_triangle_ids, path_scatter_events, path_end_type, path_sizes;
        Buffer<Float> photon_etas, path_etas;
        Buffer<Float3> path_light_ends, path_camera_starts, path_colors;
        Buffer<Float3> photon_light_starts, photon_colors;

        Buffer<UInt> path_photon_connections;
        unique_ptr<Matrix> photon_matrix, photon_matrix_param, path_matrix, path_matrix_param;
        
        
        Pipeline _pipeline;

        UInt photon_per_iter, max_photon_depth, path_per_iter, max_path_depth, max_photon_per_path;
        PhotonMappingLogger(UInt photon_per_iter, UInt max_photon_depth, UInt path_per_iter, UInt max_path_depth, UInt max_photon_per_path, Pipeline &pipeline):
            photon_per_iter(photon_per_iter), max_photon_depth(max_photon_depth), path_per_iter(path_per_iter), max_path_depth(max_path_depth), max_photon_per_path(max_photon_per_path){
            _pipeline=pipeline;
            auto &&device = pipeline.device();
            photon_inst_ids = pipeline.create<Buffer<UInt>>(photon_per_iter * max_photon_depth);
            photon_triangle_ids = pipeline.create<Buffer<UInt>>(photon_per_iter * max_photon_depth);
            path_inst_ids = pipeline.create<Buffer<UInt>>(path_per_iter * max_path_depth);
            path_triangle_ids = pipeline.create<Buffer<UInt>>(path_per_iter * max_path_depth);
            photon_etas = pipeline.create<Buffer<Float>>(photon_per_iter * max_photon_depth);
            path_etas = pipeline.create<Buffer<Float>>(path_per_iter * max_path_depth);
            path_light_ends = pipeline.create<Buffer<Float3>>(path_per_iter*max_path_depth);
            photon_light_starts = pipeline.create<Buffer<Float3>>(photon_per_iter);
            path_camera_starts = pipeline.create<Buffer<Float3>>(path_per_iter);
            photon_colors = pipeline.create<Buffer<Float3>>(photon_per_iter*max_photon_depth);
            path_photon_connections = pipeline.create<Buffer<UInt>>(path_per_iter * max_photon_per_path);

            path_sizes = pipeline.create<Buffer<UInt>>(path_per_iter);
            photon_sizes = pipeline.create<Buffer<UInt>>(photon_per_iter);
        }

        void add_start_light(UInt photon_id, LightSampler::Sample &light_sample){
            auto index = photon_id; 
            auto beta = light_sample.eval.L / light_sample.eval.pdf;
            auto start = light_sample.shadow_ray.origin;
            photon_light_starts->write(index, start);
        }

        void add_start_camera(UInt path_id, Ray* ray, Float3 beta){
            auto index = path_id; 
            path_camera_starts->write(index, ray->origin());
        }

        void add_photon_vertex(UInt photon_id, UInt path_size, Interaction* it, Surface::Sample &s, Float eta){
            auto index = photon_id * max_photon_depth + path_size;
            photon_inst_ids->write(index, it->instance_id());
            photon_triangle_ids->write(index, it->triangle_id());
            photon_etas->write(index, eta);
            photon_colors->write(index, s.eval.f);
            photon_scatter_events->write(index, s.eval.events);
        }

        void add_path_vertex(UInt path_id, UInt path_size, Interaction* it, Surface::Sample &s, Float eta){
            auto index = path_id * max_path_depth + path_size;
            path_inst_ids->write(index, it->instance_id());
            path_triangle_ids->write(index, it->triangle_id());
            path_etas->write(index, eta);
            path_colors->write(index, s.eval.f);
            path_scatter_events->write(index, s.eval.events);
        }

        void add_light_end(UInt path_id, UInt path_size, Interaction* it, Light::Evaluation*eval){
            auto index = path_id * max_path_depth + path_size;
            path_light_ends->write(index, it->p());
            path_end_type->write(index, 1u|path_end_type->read(index));
        }

        void add_envlight_end(UInt path_id, UInt path_size, Float3 end, Light::Evaluation*eval){
            auto index = path_id * max_path_depth + path_size;
            path_light_ends->write(index, end);
            path_end_type->write(index, 2u|path_end_type->read(index));
        }

        void connect_path_photon(UInt path_id, UInt photon_id, UInt path_photon_size, Float3 dis, Float3 Phi){
            auto index = path_id * max_photon_per_path + path_photon_size;
            path_photon_connections->write(index, photon_id);
        }

        void add_path_sizes(UInt path_id, UInt path_size){
            path_sizes->write(path_id, path_size);
        }

        void add_indirect_end(UInt path_id, UInt path_size, Float3 beta){
            path_indirect_ends->write(path_id, beta);
            path_end_type->write(index, 4u|path_end_type->read(index));
        }

    };

    PhotonMappingLogger logger;

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
        
        using namespace luisa::compute;

        Kernel2D view_path_gradient_compute_kernel = [&](Buffer<Float> &grad_in) noexcept {
            //EPSM compute color and position gradients w.r.t to view path.
        };

        Kernel2D emit_photons_bp_kernel = [&](UInt frame_index, Float time, Buffer<Float> &grad_in) noexcept {
            auto pixel_id = dispatch_id().xy();
            auto sampler_id = UInt2(pixel_id.x + resolution.x, pixel_id.y);
            $if(pixel_id.x * resolution.y + pixel_id.y < photon_per_iter) {
                photon_tracing_bp(viewpoints, indirect, camera, frame_index, sampler_id, time, pixel_id.x * resolution.y + pixel_id.y, grad_in);
            };
        };
        
        Kernel1D accumulate_gradients_kernel = [&]() noexcept {
            
        };

        auto view_path_gradient_compute = pipeline().device().compile(view_path_gradient_compute_kernel);
        auto emit_photons_bp = pipeline().device().compile(emit_photons_bp_kernel);
        auto accumulate_gradients = pipeline().device().compile(accumulate_gradients_kernel);
    }
    void _render_one_camera(CommandBuffer &command_buffer, Camera::Instance *camera) noexcept override {
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
        //TODO: use sampler right

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
        PixelIndirect indirect(viewpoints_per_iter, spectrum, camera->film(), clamp, node<MegakernelPhotonMappingDiff>()->shared_radius());
        ViewPointMap viewpoints(viewpoints_per_iter * node<MegakernelPhotonMappingDiff>()->max_depth(), spectrum);

        //pathlogger = make_unique<PathLogger>(node<MegakernelPhotonMappingDiff>()->max_depth(), node<MegakernelPhotonMappingDiff>()->photon_per_iter(), spectrum);
        //initialize PixelIndirect
        Kernel2D indirect_initialize_kernel = [&]() noexcept {
            
            auto index = dispatch_id().xy();
            auto radius = node<MegakernelPhotonMapping>()->initial_radius();
            if (radius < 0)
                photons.write_grid_len(photons.split(-radius));
            else
                photons.write_grid_len(node<MegakernelPhotonMapping>()->initial_radius());
            //camera->pipeline().printer().info("grid:{}", photons.grid_len());
            indirect.write_radius(index, photons.grid_len());
            //camera->pipeline().printer().info("rad:{}", indirect.radius(index));

            indirect.write_cur_n(index, 0u);
            indirect.write_n_photon(index, 0u);
            indirect.reset_phi(index);
            indirect.reset_tau(index);
        };
        
        Kernel1D viewpoint_reset_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            viewpoints.reset(index);
        };

        Kernel2D viewpath_construct_kernel = [&](UInt frame_index, Float time, Float shutter_weight) noexcept {
            //Construct view path
            //@Todo:What is set block size?
            auto pixel_id = dispatch_id().xy();
            auto L = emit_viewpoint(camera, frame_index, pixel_id, time, shutter_weight);
            camera->film()->accumulate(pixel_id, L, 0.5f);
        };

        Kernel1D build_grid_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            auto radius = node<MegakernelPhotonMappingDiff>()->initial_radius();
            $if(viewpoints.nxt(index) == 0u) {
                viewpoints.link(index);
            };
        };

        Kernel2D emit_photons_kernel = [&](UInt frame_index, Float time) noexcept {
            auto pixel_id = dispatch_id().xy();
            auto sampler_id = UInt2(pixel_id.x + resolution.x, pixel_id.y);
            $if(pixel_id.x * resolution.y + pixel_id.y < photon_per_iter) {
                photon_tracing(viewpoints, indirect, camera, frame_index, sampler_id, time);
            };
        };

        Kernel2D update_info_kernel = [&]() noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            indirect.pixel_info_update(pixel_id);
        };

        Kernel2D indirect_update_kernel = [&]() noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            indirect.pixel_info_update(pixel_id);
        };

        Kernel1D shared_update_kernel = [&]() noexcept {
            indirect.shared_update();
            photons.write_grid_len(indirect.radius(make_uint2(0, 0)));
        };

        //accumulate the stored indirect light into final image
        Kernel2D indirect_draw_kernel = [&](UInt tot_photon, UInt spp) noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_id().xy();
            auto L = get_indirect(indirect, camera->pipeline().spectrum(), pixel_id, tot_photon);
            camera->film()->accumulate(pixel_id, L, 0.5f * spp);
        };
        Clock clock_compile;

        auto indirect_initialize = pipeline().device().compile(indirect_initialize_kernel);
        auto viewpoint_reset = pipeline().device().compile(viewpoint_reset_kernel);
        auto viewpath_construct = pipeline().device().compile(viewpath_reset_kernel);
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
        
        command_buffer << indirect_initialize().dispatch(resolution) << synchronize();
        for (auto s : shutter_samples) {
            pipeline().update(command_buffer, s.point.time);
            runtime_spp += s.spp;
            for (auto i = 0u; i < s.spp; i++) {
                //emit phtons then calculate L
                //TODO: accurate size reset

                command_buffer << view_path_reset().dispatch(viewpoints.size());
                command_buffer << view_path_construct(sample_id++, s.point.time, s.point.weight).dispatch(resolution);
                command_buffer << build_grid().dispatch(resolution);
                command_buffer << emit_photon(sample_id, s.point.time).dispatch(make_uint2(add_x, resolution.y));
                command_buffer << indirect_update().dispatch(resolution);

                if (node<MegakernelPhotonMappingDiff>()->shared_radius()) {
                    command_buffer << shared_update().dispatch(1u);
                }

                dispatch_count++;
                if (camera->film()->show(command_buffer)) {
                    dispatch_count = 0u;
                }

                auto dispatches_per_commit = 4u;
                if (dispatch_count % dispatches_per_commit == 0u) [[unlikely]] {
                    dispatch_count = 0u;
                    auto p = sample_id / static_cast<double>(spp);
                    command_buffer << [&progress, p] { progress.update(p); };
                }
            }
            command_buffer << pipeline().printer().retrieve();
        }

        LUISA_INFO("total spp:{}", runtime_spp);
        // tot_photon is photon_per_iter not photon_per_iter*spp because of unnormalized samples
        command_buffer << indirect_draw(node<MegakernelPhotonMappingDiff>()->photon_per_iter(), runtime_spp).dispatch(resolution);
        command_buffer << synchronize();
        command_buffer << pipeline().printer().retrieve();
        progress.done();
        auto render_time = clock.toc();
        LUISA_INFO("Rendering finished in {} ms.", render_time);
    }

    [[nodiscard]] Float3 get_indirect(PixelIndirect &indirect, const Spectrum::Instance *spectrum, Expr<uint2> pixel_id, Expr<uint> tot_photon) noexcept {
        auto r = indirect.radius(pixel_id);
        auto tau = indirect.tau(pixel_id);
        Float3 L;
        L = tau / (tot_photon * pi * r * r);
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

        auto pixel_id_1d = pixel_id.x*resolution.y+pixel_id.y;
        logger.add_start_camera(pixel_id_1d, ray, camera_weight);
        auto path_size = 0u;

        $for(depth, node<MegakernelPhotonMappingDiff>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    logger.add_envlight_end(pixel_id_1d, path_size, ray->direction(), eval);
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    logger.add_light_end(pixel_id_1d, path_size, it, eval);
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
                
                logger.add_path_vertex(pixel_id_1d, path_size, it, light_sample.eval, 1.f);
                path_size+=1;

                // direct lighting
                $if(light_sample.eval.pdf > 0.0f & !occluded) {
                    auto wi = light_sample.shadow_ray->direction();
                    auto eval = closure->evaluate(wo, wi);
                    auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                light_sample.eval.pdf;
                    Li += w * beta * eval.f * light_sample.eval.L;
                    logger.add_light_end(pixel_id_1d, path_size, it, light_sample.eval);
                };

                auto roughness = closure->roughness();
                Bool stop_check = (roughness.x * roughness.y > 0.16f) | stop_direct;
                $if(stop_check) {
                    stop_direct = true;
                    viewpoint_map.push(it->p(), swl, beta, wo, pixel_id, closure);
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
                logger.add_indirect_end(pixel_id_1d, path_size, beta);
                // miss
                $if(!it_next->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        logger.add_envlight_end(pixel_id_1d, path_size, ray->direction(), eval);
                    }
                };
                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it_next->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it_next, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        logger.add_light_end(pixel_id_1d, path_size, it_next, eval);
                    };
                }
                $break;
            };
            $if(depth + 1u >= rr_depth) {
                beta *= ite(q < rr_threshold, 1.0f / q, 1.f);
            };
        };
        //return spectrum->srgb(swl, testbeta);//DEBUG
        return spectrum->srgb(swl, Li);
    }

    void photon_tracing_bp(ViewPointMap &viewpoints, const Camera::Instance *camera, Expr<uint> frame_index,
                        Expr<uint2> sampler_id, Expr<float> time, Expr<uint> photon_id_1d, Buffer<Float> &grad_in) {

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
        
        //@Todo: log path.
        
        auto ray = light_sample.shadow_ray;
        auto pdf_bsdf = def(1e16f);

        logger.add_start_light(photon_id_1d, light_sample);
        auto path_size = 0u; 

        ArrayUInt<12> triangle_ids, inst_ids;
        ArrayFloat3<12> bary_coords;
        ArrayFloat<12> etas;
        ArrayFloat<64x64> mat;

        $for(depth, node<MegakernelPhotonMappingDiff>()->max_depth()) {

            // trace
            auto wi = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            // miss
            $if(!it->valid()) {
                $break;
            };


            $if(!it->shape().has_surface()) { $break; };
            
            triangle_ids[path_size] = it->triangle_id();
            inst_ids[path_size] = it->instance_id();
            bary_coords[path_size] = it->bary_coord();
            //logger.add_photon_vertex(photon_id_1d, path_size, it);
            path_size+=1;
            
            // generate uniform samples
            auto u_lobe = sampler()->generate_1d();
            auto u_bsdf = sampler()->generate_2d();
            auto u_rr = def(0.f);
            auto rr_depth = node<MegakernelPhotonMappingDiff>()->rr_depth();
            $if(depth + 1u >= rr_depth) { u_rr = sampler()->generate_1d(); };
            $if(depth > 0) {// add diffuse constraint?
                auto grid = viewpoints.point_to_grid(it->p());
                
                Float3 grad_beta = make_float3(0.f);
                Float2 grad_bary = make_float2(0.f);
                UInt count_neighbors = 0;
                $for(x, grid.x - 1, grid.x + 2) {
                    $for(y, grid.y - 1, grid.y + 2) {
                        $for(z, grid.z - 1, grid.z + 2) {
                            Int3 check_grid{x, y, z};
                            auto viewpoint_index = viewpoints.grid_head(viewpoints.grid_to_index(check_grid));
                            $while(viewpoint_index != ~0u) {
                                auto position = viewpoints.position(viewpoint_index);
                                auto pixel_id = viewpoints.pixel_id(viewpoint_index);
                                auto closure = viewpoints.closure(viewpoint_index);
                                auto dis = distance(position, it->p());
                                auto rad = indirect.radius(pixel_id);
                                $if(dis <= rad) {
                                    auto viewpoint_beta = logger.read_indirect_path_beta(pixel_id);
                                    auto eval_viewpoint = closure->evaluate(wi, viewpoint_wo);
                                    auto bary = bary_coords[path_size];
                                    auto instance = pipeline().geometry()->instance(inst_ids[0]);
                                    auto triangle = pipeline().geometry()->triangle(instance, triangle_ids[0]);
                                    auto v_buffer = instance.vertex_buffer_id();
                                    auto v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
                                    auto v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
                                    auto v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);
                                    auto point_0 = v0->position();
                                    auto point_1 = v1->position();
                                    auto point_2 = v2->position();
                                    count_neighbors++;
                                    $autodiff{
                                        requires_grad(bary, beta);
                                        Float3 photon_pos = point_0*bary_pre[0]+point_1*bary_pre[1]+point_2*(1-bary_pre[0]-bary_pre[1]);
                                        auto dis_diff = distance(position, photon_pos);
                                        auto weight = spline_kernel_eval(dis_diff, rad);
                                        auto Phi = weight * viewpoint_beta * eval_viewpoint * beta;
                                        backward(Phi);
                                        grad_beta += grad(beta);
                                        grad_bary += grad(bary);
                                        //compute gradient w.r.t to photon position and power
                                    };
                                };
                                //pipeline().printer().info("check_grid:{},{},{};test_grid:{},{},{}; limit:{}", x, y, z, test_grid[0], test_grid[1], test_grid[2], indirect.radius(pixel_id));
                                // $if(dis <= indirect.radius(pixel_id)) {
                                //     auto viewpoint_wo = viewpoints.wo(photon_index);
                                //     auto viewpoint_beta = viewpoints.beta(photon_index);
                                //     auto eval_viewpoint = closure->evaluate(wi, viewpoint_wo);
                                //     auto wo_local = it->shading().world_to_local(viewpoint_wo);
                                //     Float3 Phi;
                                //     if (!spectrum->node()->is_fixed()) {
                                //         auto viewpoint_swl = viewpoints.swl(photon_index);
                                //         Phi = spectrum->wavelength_mul(swl, beta * (eval_viewpoint.f / abs_cos_theta(wo_local)), viewpoint_swl, viewpoint_beta);
                                //     } else {
                                //         Phi = spectrum->srgb(swl, beta * viewpoint_beta * eval_photon.f / abs_cos_theta(wo_local));
                                //     }

                                // };
                                viewpoint_index = viewpoints.nxt(viewpoint_index);
                            };
                        };
                    };
                };
                $if(count_neighbors>0){
                    grad_beta/=count_neighbors;
                    grad_bary/=count_neighbors;
                    EPSM_photon(inst_ids, triangle_ids, etas, light_sample, grad_beta, grad_bary);
                    //logger.photon_gradient_compute(photon_id_1d);
                    //@Todo:build matrix and solve it 
                    $break;// break?
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
                
                auto roughness = closure->roughness();
                logger.set_photon_mat(photon_id_1d, paths_size-1, surface_sample, eta);

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

    void photon_tracing(ViewPointMap &viewpoints, const Camera::Instance *camera, Expr<uint> frame_index,
                        Expr<uint2> sampler_id, Expr<float> time) {
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
                auto grid = viewpoints.point_to_grid(it->p());
                $for(x, grid.x - 1, grid.x + 2) {
                    $for(y, grid.y - 1, grid.y + 2) {
                        $for(z, grid.z - 1, grid.z + 2) {
                            Int3 check_grid{x, y, z};
                            auto viewpoint_index = viewpoints.grid_head(viewpoints.grid_to_index(check_grid));
                            $while(viewpoint_index != ~0u) {
                                auto position = viewpoints.position(viewpoint_index);
                                auto pixel_id = viewpoints.pixel_id(viewpoint_index);
                                auto closure = viewpoints.closure(viewpoint_index);
                                auto dis = distance(position, it->p());
                                //pipeline().printer().info("check_grid:{},{},{};test_grid:{},{},{}; limit:{}", x, y, z, test_grid[0], test_grid[1], test_grid[2], indirect.radius(pixel_id));
                                $if(dis <= indirect.radius(pixel_id)) {
                                    auto viewpoint_wo = viewpoints.wo(photon_index);
                                    auto viewpoint_beta = viewpoints.beta(photon_index);
                                    auto eval_viewpoint = closure->evaluate(wi, viewpoint_wo);
                                    auto wo_local = it->shading().world_to_local(viewpoint_wo);
                                    Float3 Phi;
                                    if (!spectrum->node()->is_fixed()) {
                                        auto viewpoint_swl = viewpoints.swl(photon_index);
                                        Phi = spectrum->wavelength_mul(swl, beta * (eval_viewpoint.f / abs_cos_theta(wo_local)), viewpoint_swl, viewpoint_beta);
                                    } else {
                                        Phi = spectrum->srgb(swl, beta * viewpoint_beta * eval_photon.f / abs_cos_theta(wo_local));
                                    }
                                    indirect.add_phi(pixel_id, Phi);
                                    indirect.add_cur_n(pixel_id, 1u);
                                };
                                viewpoint_index = viewpoints.nxt(viewpoint_index);
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
                };
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

    void EPSM_photon(ArrayUInt<12> inst_ids, ArrayUInt<12> triangles_ids, ArrayFloat<12> etas, std::pair<Light::Sample, Var<Ray>> light_sample, Float3 grad_beta, Float2 grad_bary){
    {
        auto instance = pipeline().geometry()->instance(inst_ids[0]);
        auto triangle = pipeline().geometry()->triangle(instance, triangle_ids[0]);
        auto v_buffer = instance.vertex_buffer_id();
        auto v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
        auto v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
        auto v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);
        
        auto point_pre_0 = v0->position();
        auto point_pre_1 = v1->position();
        auto point_pre_2 = v2->position();
        auto bary_pre = bary_coords[0];
        
        instance = pipeline().geometry()->instance(inst_ids[1]);
        triangle = pipeline().geometry()->triangle(instance, triangle_ids[1]);
        v_buffer = instance.vertex_buffer_id();
        v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
        v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
        v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);
        
        auto point_cur_0 = v0->position();
        auto point_cur_1 = v1->position();
        auto point_cur_2 = v2->position();
        auto bary_cur = bary_coords[1];

        //need bary coord

        $for(id, 1u, path_size-1){
            
            auto normal_cur_0 = v0->normal();
            auto normal_cur_1 = v1->normal();
            auto normal_cur_2 = v2->normal();
            
            instance = pipeline().geometry()->instance(inst_ids[id + 1]);
            triangle = pipeline().geometry()->triangle(instance, triangle_ids[id + 1]);
            v_buffer = instance.vertex_buffer_id();
            v0 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i0);
            v1 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i1);
            v2 = pipeline().buffer<Vertex>(v_buffer).read(triangle.i2);
            auto point_nxt_0 = v0->position();
            auto point_nxt_1 = v1->position();
            auto point_nxt_2 = v2->position();
            auto bary_nxt = bary_coords[id+1];

            $autodiff{
                requires_grad(point_pre_0, point_pre_1, point_pre_2);
                requires_grad(point_cur_0, point_cur_1, point_cur_2);
                requires_grad(point_nxt_0, point_nxt_1, point_nxt_2);
                requires_grad(bary_pre, bary_cur, bary_nxt);

                Float3 point_pre = point_pre_0*bary_pre[0]+point_pre_1*bary_pre[1]+point_pre_2*(1-bary_pre[0]-bary_pre[1]);
                Float3 point_cur = point_cur_0*bary_cur[0]+point_cur_1*bary_cur[1]+point_cur_2*(1-bary_cur[0]-bary_cur[1]);
                Float3 point_nxt = point_nxt_0*bary_nxt[0]+point_nxt_1*bary_nxt[1]+point_nxt_2*(1-bary_nxt[0]-bary_nxt[1]);
                Float3 normal_cur = normal_cur_0*bary_cur[0]+normal_cur_1*bary_cur[1]+normal_cur_2*(1-bary_cur[0]-bary_cur[1]);

                auto trans_mat = create_local_frame(normal_cur);
                auto wi = normalize(point_pre-point_cur);
                auto wo = normalize(point_nxt-point_cur);
                auto wi_local = trans_mat*wi;
                auto wo_local = trans_mat*wo;
                auto res = normalize(wi_local+wo_local*etas[id]);   

                //auto res = wi + wo;
                //Todo Clear Grad

                for(int j=0;j<2;j++)
                {
                    backward(res[j]);
                    auto grad_uv_pre = grad(bary_pre);
                    auto grad_uv_cur = grad(bary_cur);
                    auto grad_uv_nxt = grad(bary_nxt);
                    mat->set(pixel_id_1d, id*2-2+j, 2*id-2, grad_uv_pre[0]);
                    mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id - 1, grad_uv_pre[1]);
                    mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id - 0, grad_uv_cur[0]);
                    mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 1, grad_uv_cur[1]);
                    mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 2, grad_uv_nxt[0]);
                    mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 3, grad_uv_nxt[1]);
                    auto point_pre_0_grad = grad(point_pre_0);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 0, point_pre_0_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 1, point_pre_0_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 2, point_pre_0_grad[2]);
                    auto point_pre_1_grad = grad(point_pre_1);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 3, point_pre_1_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 4, point_pre_1_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 5, point_pre_1_grad[2]);
                    auto point_pre_2_grad = grad(point_pre_2);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 6, point_pre_2_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 7, point_pre_2_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 8, point_pre_2_grad[2]);

                    auto point_nxt_0_grad = grad(point_nxt_0);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 0, point_nxt_0_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 1, point_nxt_0_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 2, point_nxt_0_grad[2]);
                    auto point_nxt_1_grad = grad(point_nxt_1);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 3, point_nxt_1_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 4, point_nxt_1_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 5, point_nxt_1_grad[2]);
                    auto point_nxt_2_grad = grad(point_nxt_2);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 6, point_nxt_2_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 7, point_nxt_2_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 8, point_nxt_2_grad[2]);

                    auto point_cur_0_grad = grad(point_cur_0);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 0, point_cur_0_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 1, point_cur_0_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 2, point_cur_0_grad[2]);
                    auto point_cur_1_grad = grad(point_cur_1);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 3, point_cur_1_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 4, point_cur_1_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 5, point_cur_1_grad[2]);
                    auto point_cur_2_grad = grad(point_cur_2);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 6, point_cur_2_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 7, point_cur_2_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 8, point_cur_2_grad[2]);
                    
                    auto normal_cur_0_grad = grad(point_cur_0);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 9, normal_cur_0_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 10, normal_cur_0_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 11, normal_cur_0_grad[2]);
                    auto normal_cur_1_grad = grad(point_cur_1);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 12, normal_cur_1_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 13, normal_cur_1_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 14, normal_cur_1_grad[2]);
                    auto normal_cur_2_grad = grad(point_cur_2);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 15, normal_cur_2_grad[0]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 16, normal_cur_2_grad[1]);
                    mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 17, normal_cur_2_grad[2]);
                }
            };
            point_pre_0 = point_cur_0;
            point_pre_1 = point_cur_1;
            point_pre_2 = point_cur_2;
            bary_pre = bary_cur;
            point_cur_0 = point_nxt_0;
            point_cur_1 = point_nxt_1;
            point_cur_2 = point_nxt_2;
            bary_cur = bary_nxt;
        };
        inverse_matrix(pixel_id_1d, path_size, max_depth);
        compute_and_scatter_grad(pixel_id_1d, path_size, path_size * 6, inst_ids, triangle_ids);
    }
};

luisa::unique_ptr<Integrator::Instance> MegakernelPhotonMappingDiff::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<MegakernelPhotonMappingDiffInstance>(
        pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MegakernelPhotonMappingDiff)
