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
class MegakernelPhotonMappingDiff: public DifferentiableIntegrator {

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
    
    class PhotonMappingLogger{
    public:
        //Buffer<uint> photon_inst_ids, photon_triangle_ids, photon_scatter_events, photon_sizes;
        Buffer<uint> path_inst_ids, path_triangle_ids, path_scatter_events, path_end_type, path_sizes, path_light_insts, path_light_tris;
        Buffer<float> path_etas, indirect_betas, path_betas, path_camera_weights,path_light_colors;
        Buffer<float3> path_camera_starts, path_light_ends;

        Buffer<float3> path_barys, path_light_barys;



        // Buffer<float3> photon_light_starts, photon_colors;
        // Buffer<float> photon_etas;
        //Buffer<UInt> path_photon_connections;
        //unique_ptr<Matrix> photon_matrix, photon_matrix_param, path_matrix, path_matrix_param;
        //Pipeline _pipeline;
        uint path_per_iter, max_path_depth;
        //uint photon_per_iter, max_photon_depth;
        const Spectrum::Instance *_spectrum;
        const uint _dimension;
        PhotonMappingLogger(uint path_per_iter, uint max_path_depth, const Spectrum::Instance *spectrum) :
            path_per_iter(path_per_iter), max_path_depth(max_path_depth),
            _spectrum(spectrum),_dimension(spectrum->node()->dimension()){
            auto &&device = spectrum->pipeline().device();
            //photon_inst_ids = device.create_buffer<uint>(photon_per_iter * max_photon_depth);
            //photon_triangle_ids = device.create_buffer<uint>(photon_per_iter * max_photon_depth);
            //photon_etas = device.create_buffer<float>(photon_per_iter * max_photon_depth);
            //photon_light_starts = device.create_buffer<float3>(photon_per_iter);
            //photon_colors = device.create_buffer<float3>(photon_per_iter * max_photon_depth);
            //photon_sizes = device.create_buffer<uint>(photon_per_iter);
            //path_photon_connections = device.create_buffer<UInt>>(path_per_iter * max_photon_per_path);
            path_inst_ids = device.create_buffer<uint>(path_per_iter * max_path_depth);
            path_triangle_ids = device.create_buffer<uint>(path_per_iter * max_path_depth);
            path_scatter_events = device.create_buffer<uint>(path_per_iter * max_path_depth);
            path_etas = device.create_buffer<float>(path_per_iter * max_path_depth);
            path_barys = device.create_buffer<float3>(path_per_iter * max_path_depth);

            path_light_ends = device.create_buffer<float3>(path_per_iter * max_path_depth);
            path_light_insts = device.create_buffer<uint>(path_per_iter * max_path_depth);
            path_light_tris = device.create_buffer<uint>(path_per_iter * max_path_depth);
            path_light_barys = device.create_buffer<float3>(path_per_iter * max_path_depth);
            path_end_type = device.create_buffer<uint>(path_per_iter * max_path_depth);

            path_camera_weights = device.create_buffer<float>(path_per_iter);
            path_camera_starts = device.create_buffer<float3>(path_per_iter);

            path_sizes = device.create_buffer<uint>(path_per_iter);
            indirect_betas = device.create_buffer<float>(path_per_iter * _dimension);
            path_betas = device.create_buffer<float>(path_per_iter * max_path_depth * _dimension);
            path_light_colors = device.create_buffer<float>(path_per_iter * max_path_depth * _dimension);
        }

        void add_start_camera(Expr<uint> path_id, Var<Ray> &ray, Float beta) {
            path_camera_starts->write(path_id, ray->origin());
            path_camera_weights->write(path_id, beta);
        }

        void add_path_vertex(Expr<uint> path_id, Expr<uint> path_size, shared_ptr<Interaction> it) {
            auto index = path_id * max_path_depth + path_size;
            path_inst_ids->write(index, it->instance_id());
            path_triangle_ids->write(index, it->triangle_id());
            path_barys->write(index, it->bary_coord());
        }

        void set_path_vertex_attrib(Expr<uint> path_id, Expr<uint> path_size, Surface::Sample &s, Float eta) {
            auto index = path_id * max_path_depth + path_size;
            path_etas->write(index, eta);
            path_scatter_events->write(index, s.event);
            for (auto i = 0u; i < _dimension; ++i)
                path_betas->write(index * _dimension + i, s.eval.f[i]);
        }

        void add_light_end(Expr<uint> path_id, Expr<uint> path_size, shared_ptr<Interaction> it, SampledSpectrum L) {
            //return;
            auto index = path_id * max_path_depth + path_size;
            path_light_ends->write(index, it->p());
            path_light_insts->write(index, it->instance_id());
            path_light_tris->write(index, it->triangle_id());
            path_light_barys->write(index, it->bary_coord());
            for (auto i = 0u; i < _dimension; ++i)
                path_light_colors->write(index * _dimension + i, L[i]);
            path_end_type->write(index, 1u|path_end_type->read(index));
        }

        void add_envlight_end(Expr<uint> path_id, Expr<uint> path_size, Float3 end, SampledSpectrum L) {
            //return;
            auto index = path_id * max_path_depth + path_size;
            path_light_ends->write(index, end);
            for (auto i = 0u; i < _dimension; ++i)
                path_light_colors->write(index * _dimension + i, L[i]);
            path_end_type->write(index, 2u|path_end_type->read(index));
        }

        /*
        void add_start_light(Expr<uint> photon_id, LightSampler::Sample &light_sample) {
            auto index = photon_id;
            auto beta = light_sample.eval.L / light_sample.eval.pdf;
            auto start = light_sample.shadow_ray->origin();
            photon_light_starts->write(index, start);
        }
        void add_photon_vertex(Expr<uint> photon_id, Expr<uint> path_size, shared_ptr<Interaction> it, Surface::Sample &s, Float eta) {
            auto index = photon_id * max_photon_depth + path_size;
            photon_inst_ids->write(index, it->instance_id());
            photon_triangle_ids->write(index, it->triangle_id());
            photon_etas->write(index, eta);
            photon_colors->write(index, s.eval.f);
            //photon_scatter_events->write(index, s.eval.events);
        }
        void connect_path_photon(UInt path_id, UInt photon_id, UInt path_photon_size, Float3 dis, Float3 Phi){
            auto index = path_id * max_photon_per_path + path_photon_size;
            path_photon_connections->write(index, photon_id);
        }

        void add_path_sizes(UInt path_id, UInt path_size){
            path_sizes->write(path_id, path_size);
        }

        void add_indirect_end(Expr<uint> path_id, Expr<uint> path_size, SampledSpectrum beta) {
            for (auto i = 0u; i < _dimension; ++i)
                indirect_betas->write(path_id * _dimension + i, beta[i]);
            //path_end_type->write(index, 4u|path_end_type->read(index));
        }

        auto indirect_beta(Expr<uint> path_id) {
            SampledSpectrum s{_dimension};
            for (auto i = 0u; i < _dimension; ++i)
                s[i] = indirect_betas->read(path_id * _dimension + i);
            return s;
        }
        */

    };

    luisa::unique_ptr<PhotonMappingLogger> logger;
    luisa::unique_ptr<ViewPointMap> viewpoints;
    luisa::unique_ptr<PixelIndirect> indirect;

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
        
        logger = make_unique<PhotonMappingLogger>(pixel_count, node<MegakernelPhotonMappingDiff>()->max_depth(), spectrum);
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
            auto L = emit_viewpoint_bp(camera, frame_index, pixel_id, time, shutter_weight);
            camera->film()->accumulate(pixel_id, L, 0.5f);
        };

        Kernel1D build_grid_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            auto radius = node<MegakernelPhotonMappingDiff>()->initial_radius();
            $if(viewpoints->nxt(index) == 0u) {
                viewpoints->link(index);
            };
        };

        Kernel2D emit_photons_kernel = [&](UInt frame_index, Float time, BufferFloat grad_in) noexcept {
            auto pixel_id = dispatch_id().xy();
            auto sampler_id = UInt2(pixel_id.x + resolution.x, pixel_id.y);
            $if(pixel_id.x * resolution.y + pixel_id.y < photon_per_iter) {
                photon_tracing_bp(camera, frame_index, sampler_id, time, pixel_id.x * resolution.y + pixel_id.y, grad_in);
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
            auto pixel_id_1d = pixel_id.x * resolution.y + pixel_id.y;
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
                command_buffer << emit_photon(sample_id++, s.point.time, grad_in).dispatch(make_uint2(add_x, resolution.y));
                command_buffer << indirect_update().dispatch(viewpoints_per_iter);
                if (node<MegakernelPhotonMappingDiff>()->shared_radius()) {
                    command_buffer << shared_update().dispatch(1u);
                }
            }
        }
        command_buffer << indirect_draw(node<MegakernelPhotonMappingDiff>()->photon_per_iter(), runtime_spp).dispatch(resolution);
        LUISA_INFO("Finishi indirect_draw");
        command_buffer << synchronize();
        command_buffer << pipeline().printer().retrieve();
        progress.done();
        auto render_time = clock.toc();
        LUISA_INFO("Backward Rendering finished in {} ms.", render_time);
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
        
        logger = make_unique<PhotonMappingLogger>(pixel_count, node<MegakernelPhotonMappingDiff>()->max_depth(), spectrum);
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
            $if(pixel_id.x * resolution.y + pixel_id.y < photon_per_iter) {
                photon_tracing(camera, frame_index, sampler_id, time, pixel_id.x * resolution.y + pixel_id.y);
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
            auto pixel_id_1d = pixel_id.x * resolution.y + pixel_id.y;
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
        LUISA_INFO("Finishi indirect_draw");
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
        auto pixel_id_1d = pixel_id.x*resolution.y+pixel_id.y;

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
        //return spectrum->srgb(swl, testbeta);//DEBUG
        return spectrum->srgb(swl, Li);
    }
    
    void photon_tracing(const Camera::Instance *camera, Expr<uint> frame_index,
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
                                    indirect->add_cur_w(pixel_id, weight);
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
        auto pixel_id_1d = pixel_id.x*resolution.y+pixel_id.y;
        logger->add_start_camera(pixel_id_1d, ray, camera_weight);
        auto path_size = 0u;
        $for(depth, node<MegakernelPhotonMappingDiff>()->max_depth()) {

            // trace
            auto wo = -ray->direction();
            auto it = pipeline().geometry()->intersect(ray);

            $if(!it->valid()) {
                if (pipeline().environment()) {
                    auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    logger->add_envlight_end(pixel_id_1d, path_size, ray->direction(), eval.L);
                }
                $break;
            };

            // hit light
            if (!pipeline().lights().empty()) {
                $if(it->shape().has_light()) {
                    auto eval = light_sampler()->evaluate_hit(*it, ray->origin(), swl, time);
                    Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                    logger->add_light_end(pixel_id_1d, path_size, it, eval.L);
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
                logger->add_path_vertex(pixel_id_1d, path_size, it);
                // direct lighting
                $if(light_sample.eval.pdf > 0.0f & !occluded) {
                    auto wi = light_sample.shadow_ray->direction();
                    auto eval = closure->evaluate(wo, wi);
                    auto w = balance_heuristic(light_sample.eval.pdf, eval.pdf) /
                                light_sample.eval.pdf;
                    Li += w * beta * eval.f * light_sample.eval.L;
                    logger->add_light_end(pixel_id_1d, path_size, it, light_sample.eval.L);
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

                logger->set_path_vertex_attrib(pixel_id_1d, path_size, surface_sample, eta);
                path_size += 1;
                
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
                $if(!it_next->valid()) {
                    if (pipeline().environment()) {
                        auto eval = light_sampler()->evaluate_miss(ray->direction(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        logger->add_envlight_end(pixel_id_1d, path_size, ray->direction(), eval.L);
                    }
                };
                // hit light
                if (!pipeline().lights().empty()) {
                    $if(it_next->shape().has_light()) {
                        auto eval = light_sampler()->evaluate_hit(*it_next, ray->origin(), swl, time);
                        Li += beta * eval.L * balance_heuristic(pdf_bsdf, eval.pdf);
                        logger->add_light_end(pixel_id_1d, path_size, it_next, eval.L);
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

    void photon_tracing_bp(const Camera::Instance *camera, Expr<uint> frame_index,
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
        auto path_size = 0u; 

        ArrayUInt<4> triangle_ids, inst_ids;
        ArrayFloat3<4> bary_coords, points, normals;
        ArrayFloat<4> etas;
        ArrayFloat2<4> grad_barys;
        ArrayFloat3<4> grad_betas;
        
        ArrayFloat<8*8*2> mat_bary;
        ArrayFloat3<4*4> mat_param;

        auto max_depth = min(node<MegakernelPhotonMappingDiff>()->max_depth(), 4u);
        auto tot_neighbors = 0u;
        $for(depth, max_depth) {
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
            points[path_size] = it->p();
            normals[path_size] = it->ng();
            path_size+=1;
            
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
                                    auto surface_tag = viewpoints->surface_tag(viewpoint_index);
                                    SampledSpectrum eval_viewpoint(3u);
                                    PolymorphicCall<Surface::Closure> call;
                                    pipeline().surfaces().dispatch(surface_tag, [&](auto surface) noexcept {
                                        surface->closure(call, *it, swl, viewpoint_wo, 1.f, time);
                                    });
                                    call.execute([&](const Surface::Closure *closure) noexcept {
                                        eval_viewpoint = closure->evaluate(viewpoint_wo, wi).f; 
                                    });
                                    $autodiff {
                                        Float3 beta_diff = make_float3(beta[0u], beta[1u], beta[2u]);
                                        requires_grad(bary, beta_diff);
                                        Float3 photon_pos = point_0 * bary[0] + point_1 * bary[1] + point_2 * (1 - bary[0] - bary[1]);
                                        auto rel_dis_diff = distance(position, photon_pos) / rad;
                                        auto weight = 3.5f*(1- 6*pow(rel_dis_diff, 5.) + 15*pow(rel_dis_diff, 4.) - 10*pow(rel_dis_diff, 3.));
                                        auto wi_local = it->shading().world_to_local(wi);
                                        auto Phi = spectrum->srgb(swl, viewpoint_beta * eval_viewpoint / abs_cos_theta(wi_local));
                                        auto Phi_beta = Phi * beta_diff * weight;
                                        auto _grad_dimension = 3u;
                                        auto grad_pixel_0 = grad_in->read(pixel_id*_grad_dimension+0);
                                        auto grad_pixel_1 = grad_in->read(pixel_id * _grad_dimension + 1);
                                        auto grad_pixel_2 = grad_in->read(pixel_id * _grad_dimension + 2);
                                        auto dldPhi = (Phi_beta[0]*grad_pixel_0 + Phi_beta[1]*grad_pixel_1 + Phi_beta[2]*grad_pixel_2) / indirect->cur_w(pixel_id);
                                        backward(dldPhi);
                                        grad_bary += grad(bary).xy();
                                        grad_beta += grad(beta_diff);
                                    };
                                    count_neighbors+=1;
                                };
                                viewpoint_index = viewpoints->nxt(viewpoint_index);
                            };
                        };
                    };
                };
                $if(count_neighbors>0){
                    tot_neighbors+=count_neighbors;
                    grad_betas[path_size] = make_float3(grad_beta[0]/count_neighbors, grad_beta[1]/count_neighbors, grad_beta[2]/count_neighbors);
                    grad_barys[path_size] = make_float2(grad_bary[0]/count_neighbors, grad_bary[1]/count_neighbors);
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
        $if(tot_neighbors>0)
        //compute gradient w.r.t to photon position and power
        {
            EPSM_photon(path_size, points, normals, inst_ids, triangle_ids, bary_coords, etas, light_sample, grad_betas, grad_barys, mat_bary, mat_param);
        };
    }
   
    void EPSM_photon(UInt path_size, ArrayFloat3<4> &points, ArrayFloat3<4> &normals, ArrayUInt<4> &inst_ids, ArrayUInt<4> &triangle_ids, ArrayFloat3<4> &bary_coords, 
    ArrayFloat<4> &etas, LightSampler::Sample &light_sample, ArrayFloat3<4> grad_beta, ArrayFloat2<4> grad_bary, ArrayFloat<8 * 8 * 2> &mat_bary, ArrayFloat3<4 * 4> &mat_param){
    {
        // change it to light sample: if env light use A else Use b
        Callable create_local_frame = [](Float3 x) {
            auto normal = normalize(x);
            auto tangent = normalize(cross(normal, Float3(0,1,0)));
            auto bitangent = normalize(cross(normal, tangent));
            return Float3x3(tangent, bitangent, normal);
        };
        
        // Shared<float> *mat_bary = new Shared<float>(16*16);
        // Shared<float3> *mat_param = new Shared<float3>(32);

        auto locate = [&](UInt i, UInt j) {
            return (i<<5|j);
        };
        auto locate_adj = [&](UInt i, UInt j) {
            return (i<<5|16|j);
        };

        Callable inverse_matrix = [&]() {
            //inverse a matrix which mat->read(i,j) gives the (i,j) element
            auto n = path_size*2;
            $for (i,n) {
                mat_bary[locate_adj(i,i)] = 1;
            };
            $for (i, n) {
                $if (mat_bary[locate(i,i)]==0) {
                    auto p=n*2;
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
                        $if(factor==0) {$break;};
                        $for (k,i,n) {
                            //mat->set(pixel_id, j, k, mat->get(pixel_id, j, k) - factor * mat->get(pixel_id, i, k));
                            mat_bary[locate(j,k)] = mat_bary[locate(j,k)] - factor * mat_bary[locate(i,k)];
                        };
                        
                        $for (k,n) {
                            mat_bary[locate_adj(j,k)] = mat_bary[locate_adj(j,k)] - factor * mat_bary[locate_adj(i,k)];
                            //mat->set(pixel_id, j, k, mat->get(pixel_id, j, k+max_size*2) - factor * mat->get(pixel_id, i, k+max_size*2));
                        };
                    };
                };
            };
            $for(i, n) {
                auto f = mat_bary[locate(i,i)];
                $for(j, n) {
                    mat_bary[locate(i,j)] /= f;
                };
            };
        };
        Callable compute_and_scatter_grad = [&](){
            ArrayFloat<24> tmp;
            $for(i, path_size*2){
                tmp[i] = 0.0f;
                $for(j, path_size){
                    tmp[i] -= grad_bary[j][0] * mat_bary[locate(j * 2, i)] + grad_bary[j][1] * mat_bary[locate(j * 2 + 1, i)];
                };
            };
            $for(i, path_size) {
                Float3 grad_vertex = make_float3(0.0f), grad_normal = make_float3(0.0f);
                grad_vertex = tmp[i*2+0]*mat_param[((i*2)<<2)+1] + tmp[i*2+1]*mat_param[((i*2+1)<<2)+1] + tmp[i*2+2]*mat_param[((i*2+2)<<2)] + tmp[i*2+3]*mat_param[((i*2+3)<<2)];
                grad_normal = tmp[i*2+0]*mat_param[((i*2)<<2)+3] + tmp[i*2+1]*mat_param[((i*2+1)<<2)+3];
                $if(i > 0){
                    grad_vertex += tmp[i * 2 - 1] * mat_param[((i * 2 - 1) << 2) + 2] + tmp[i * 2 - 2] * mat_param[((i * 2 - 2) << 2) + 2];
                };
                pipeline().differentiation()->add_geom_gradients(grad_vertex, grad_normal, bary_coords[i], inst_ids[i], triangle_ids[i]);
                
                // pipeline().differentiation()->add_geom_gradients(grad_tmp, inst_ids[(UInt)(i/6)], triangle_ids[(UInt)(i/6)], i%6);
                // pipeline().differentiation()->add_geom_gradients(grad_tmp, inst_ids[(UInt)(i/6)], triangle_ids[(UInt)(i/6)], i%6);
                // pipeline().differentiation()->add_geom_gradients(grad_tmp, inst_ids[(UInt)(i/6)], triangle_ids[(UInt)(i/6)], i%6);
            };
        };

        // Float3 point_pre = light_sample.second->origin();
        // Float2 bary_pre = Float2(0.0f);
        // Float2 bary_cur = bary_coords[0];
        // Float3 point_cur = points[0];
        // Float3 point_nxt,normal_cur, bary_nxt;

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
        
        auto point_cur_0 = v0->position();
        auto point_cur_1 = v1->position();
        auto point_cur_2 = v2->position();
        auto bary_cur = bary_coords[0];

        $for(id, path_size-1){
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
            auto bary_nxt = bary_coords[id+1u];
            // point_nxt = points[id+1];
            // bary_nxt = bary_coords[id+1];
            // normal_cur = normals[id];
            $autodiff {
            // requires_grad(point_pre_0, point_pre_1, point_pre_2);
            // requires_grad(point_cur_0, point_cur_1, point_cur_2);
            // requires_grad(point_nxt_0, point_nxt_1, point_nxt_2);
                Float3 point_pre,point_cur,point_nxt,normal_cur;
                $if(id > 0) {
                    point_pre = point_pre_0 * bary_pre[0] + point_pre_1 * bary_pre[1] + point_pre_2 * (1 - bary_pre[0] - bary_pre[1]);
                } $else {
                    point_pre = light_sample.shadow_ray->origin();
                };
                point_cur = point_cur_0*bary_cur[0]+point_cur_1*bary_cur[1]+point_cur_2*(1-bary_cur[0]-bary_cur[1]);
                point_nxt = point_nxt_0*bary_nxt[0]+point_nxt_1*bary_nxt[1]+point_nxt_2*(1-bary_nxt[0]-bary_nxt[1]);
                normal_cur = normal_cur_0*bary_cur[0]+normal_cur_1*bary_cur[1]+normal_cur_2*(1-bary_cur[0]-bary_cur[1]);
                requires_grad(point_pre, point_cur, point_nxt, normal_cur);
                requires_grad(bary_pre, bary_cur, bary_nxt);

                auto trans_mat = create_local_frame(normal_cur);
                auto wi = normalize(point_pre-point_cur);
                auto wo = normalize(point_nxt-point_cur);
                auto wi_local = trans_mat*wi;
                auto wo_local = trans_mat*wo;
                auto res = normalize(wi_local+wo_local*etas[id]);   

                for(int j=0;j<2;j++)
                {
                    backward(res[j]);
                    $if(id>0)
                    {
                        // mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id - 3, grad_uv_pre[0]);
                        // mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id - 2, grad_uv_pre[1]);
                        auto grad_uv_pre = grad(bary_pre);
                        mat_bary[((((id)<<1)|j)<<4)|(((id)<<1)-2)] = grad_uv_pre[0];
                        mat_bary[((((id)<<1)|j)<<4)|(((id)<<1)-1)] = grad_uv_pre[1];
                    };
                    auto grad_uv_cur = grad(bary_cur);
                    auto grad_uv_nxt = grad(bary_nxt);
                    
                    mat_bary[((((id)<<1)|j)<<4)|(((id)<<1)+0)] = grad_uv_cur[0];
                    mat_bary[((((id)<<1)|j)<<4)|(((id)<<1)+1)] = grad_uv_cur[1];
                    
                    mat_bary[((((id)<<1)|j)<<4)|(((id)<<1)+2)] = grad_uv_nxt[0];
                    mat_bary[((((id)<<1)|j)<<4)|(((id)<<1)+3)] = grad_uv_nxt[1];

                    // mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id - 1, grad_uv_cur[0]);
                    // mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 0, grad_uv_cur[1]);
                    // mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 1, grad_uv_nxt[0]);
                    // mat->set(pixel_id_1d, id * 2 - 2 + j, 2 * id + 2, grad_uv_nxt[1]);
                    // $if(id>0){
                    // };

                    mat_param[((((id)<<1)|j)<<2)] = grad(point_pre);
                    mat_param[((((id)<<1)|j)<<2)|1] = grad(point_cur);
                    mat_param[((((id)<<1)|j)<<2)|2] = grad(point_nxt);
                    mat_param[((((id)<<1)|j)<<2)|3] = grad(normal_cur);
                    // auto point_pre_0_grad = grad(point_pre_0);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 0, point_pre_0_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 1, point_pre_0_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 2, point_pre_0_grad[2]);
                    // auto point_pre_1_grad = grad(point_pre_1);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 3, point_pre_1_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 4, point_pre_1_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 5, point_pre_1_grad[2]);
                    // auto point_pre_2_grad = grad(point_pre_2);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 6, point_pre_2_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 7, point_pre_2_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id - 1) + 8, point_pre_2_grad[2]);

                    // auto point_nxt_0_grad = grad(point_nxt_0);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 0, point_nxt_0_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 1, point_nxt_0_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 2, point_nxt_0_grad[2]);
                    // auto point_nxt_1_grad = grad(point_nxt_1);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 3, point_nxt_1_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 4, point_nxt_1_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 5, point_nxt_1_grad[2]);
                    // auto point_nxt_2_grad = grad(point_nxt_2);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 6, point_nxt_2_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 7, point_nxt_2_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id + 1) + 8, point_nxt_2_grad[2]);

                    // auto point_cur_0_grad = grad(point_cur_0);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 0, point_cur_0_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 1, point_cur_0_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 2, point_cur_0_grad[2]);
                    // auto point_cur_1_grad = grad(point_cur_1);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 3, point_cur_1_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 4, point_cur_1_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 5, point_cur_1_grad[2]);
                    // auto point_cur_2_grad = grad(point_cur_2);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 6, point_cur_2_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 7, point_cur_2_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 8, point_cur_2_grad[2]);
                    
                    // auto normal_cur_0_grad = grad(point_cur_0);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 9, normal_cur_0_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 10, normal_cur_0_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 11, normal_cur_0_grad[2]);
                    // auto normal_cur_1_grad = grad(point_cur_1);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 12, normal_cur_1_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 13, normal_cur_1_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 14, normal_cur_1_grad[2]);
                    // auto normal_cur_2_grad = grad(point_cur_2);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 15, normal_cur_2_grad[0]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 16, normal_cur_2_grad[1]);
                    // mat_param->set(pixel_id_1d, id * 2 - 2 + j, 18 * (id) + 17, normal_cur_2_grad[2]);
                }
            };
            bary_pre = bary_cur;
            bary_cur = bary_nxt;
            point_pre_0 = point_cur_0;
            point_pre_1 = point_cur_1;
            point_pre_2 = point_cur_2;
            point_cur_0 = point_nxt_0;
            point_cur_1 = point_nxt_1;
            point_cur_2 = point_nxt_2;
        };
        inverse_matrix();
        compute_and_scatter_grad();
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
