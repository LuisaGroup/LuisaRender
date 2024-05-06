// This file exports LuisaRender functionalities to a python library using pybind11.

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include "dlpack.h"

#include <span>
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <cxxopts.hpp>

#include <core/stl/format.h>
#include <core/logging.h>
#include <util/sampling.h>

#include <base/filter.h>
#include <base/scene.h>
#include <base/camera.h>
#include <base/pipeline.h>

#include <sdl/scene_parser.h>
#include <sdl/scene_desc.h>
#include <sdl/scene_node_desc.h>
#include <sdl/scene_parser_json.h>


using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

[[nodiscard]] auto parse_cli_options(int argc, const char *const *argv) noexcept {
    cxxopts::Options cli{"luisa-render-cli"};
    cli.add_option("", "b", "backend", "Compute backend name", cxxopts::value<luisa::string>(), "<backend>");
    cli.add_option("", "d", "device", "Compute device index", cxxopts::value<int32_t>()->default_value("-1"), "<index>");
    cli.add_option("", "", "scene", "Path to scene description file", cxxopts::value<std::filesystem::path>(), "<file>");
    cli.add_option("", "D", "define", "Parameter definitions to override scene description macros.",
                   cxxopts::value<std::vector<luisa::string>>()->default_value("<none>"), "<key>=<value>");
    cli.add_option("", "h", "help", "Display this help message", cxxopts::value<bool>()->default_value("false"), "");
    cli.allow_unrecognised_options();
    cli.positional_help("<file>");
    cli.parse_positional("scene");
    auto options = [&] {
        try {
            return cli.parse(argc, argv);
        } catch (const std::exception &e) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to parse command line arguments: {}.",
                e.what());
            std::cout << cli.help() << std::endl;
            exit(-1);
        }
    }();
    if (options["help"].as<bool>()) {
        std::cout << cli.help() << std::endl;
        exit(0);
    }
    if (options["scene"].count() == 0u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION("Scene file not specified.");
        std::cout << cli.help() << std::endl;
        exit(-1);
    }
    if (auto unknown = options.unmatched(); !unknown.empty()) [[unlikely]] {
        luisa::string opts{unknown.front()};
        for (auto &&u : luisa::span{unknown}.subspan(1)) {
            opts.append("; ").append(u);
        }
        LUISA_WARNING_WITH_LOCATION(
            "Unrecognized options: {}", opts);
    }
    return options;
}

[[nodiscard]] auto parse_cli_macros(int &argc, char *argv[]) {
    SceneParser::MacroMap macros;

    auto parse_macro = [&macros](luisa::string_view d) noexcept {
        if (auto p = d.find('='); p == luisa::string::npos) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Invalid definition: {}", d);
        } else {
            auto key = d.substr(0, p);
            auto value = d.substr(p + 1);
            LUISA_VERBOSE_WITH_LOCATION("Parameter definition: {} = '{}'", key, value);
            if (auto iter = macros.find(key); iter != macros.end()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Duplicate definition: {} = '{}'. "
                    "Ignoring the previous one: {} = '{}'.",
                    key, value, key, iter->second);
                iter->second = value;
            } else {
                macros.emplace(key, value);
            }
        }
    };
    // parse all options starting with '-D' or '--define'
    for (int i = 1; i < argc; i++) {
        auto arg = luisa::string_view{argv[i]};
        std::cout<<"arg "<<arg<<std::endl;
        if (arg == "-D" || arg == "--define") {
            if (i + 1 == argc) {
                LUISA_WARNING_WITH_LOCATION(
                    "Missing definition after {}.", arg);
                // remove the option
                argv[i] = nullptr;
            } else {
                parse_macro(argv[i + 1]);
                // remove the option and its argument
                argv[i] = nullptr;
                argv[++i] = nullptr;
            }
        } else if (arg.starts_with("-D")) {
            parse_macro(arg.substr(2));
            // remove the option
            argv[i] = nullptr;
        }
    }
    // remove all nullptrs
    auto new_end = std::remove(argv, argv + argc, nullptr);
    argc = static_cast<int>(new_end - argv);
    return macros;
}

class ScenePython{
public:
    luisa::unique_ptr<Scene> _scene;
    luisa::unique_ptr<Device> _device;
    luisa::unique_ptr<Pipeline> _pipeline;
    luisa::unique_ptr<Stream> _stream;
}scene_python;

class ParamStruct{
public:
    std::string type;
    uint id;
    uint size;
    uint64_t buffer_ptr;
    float4 value;
    ParamStruct(){
        type = "unknown";
        id = 0;
        size = 0;
        buffer_ptr = 0;
        value = float4(0.0f, 0.0f, 0.0f, 0.0f);
    };
    ParamStruct(std::string type, uint id, uint size, uint64_t buffer_ptr, float4 value): 
    type(type), id(id), size(size), 
    buffer_ptr(buffer_ptr), value(value){}
};

class ViewPointMap {
public:
    Buffer<uint> _grid_head;
    Buffer<float3> _beta3;
    Buffer<float3> _wo;
    Buffer<float3> _position;
    Buffer<float3> _grad_pos;
    Buffer<uint> _pixel_id;
    Buffer<uint> _surface_tags;
    Buffer<uint> _nxt;
    uint _size;//current viewpoint count
    const Spectrum::Instance *_spectrum;
    uint _dimension;
    Buffer<float> _grid_min;//atomic float3
    Buffer<float> _grid_max;//atomic float3
    Buffer<float> _grid_len;//the length of a single grid (float1)
    Buffer<uint> _tot;
    Buffer<uint> tot_test;
    float radius;
public:
    ViewPointMap(uint viewpointcount, const Spectrum::Instance *spectrum, float radius) {
        auto &&device = spectrum->pipeline().device();

        _grid_head = device.create_buffer<uint>(viewpointcount);
        _beta3 = device.create_buffer<float3>(viewpointcount);
        _wo = device.create_buffer<float3>(viewpointcount);
        _position = device.create_buffer<float3>(viewpointcount);
        _pixel_id = device.create_buffer<uint>(viewpointcount);
        _surface_tags = device.create_buffer<uint>(viewpointcount);
        _grad_pos = device.create_buffer<float3>(viewpointcount);
        _nxt = device.create_buffer<uint>(viewpointcount);
        _tot = device.create_buffer<uint>(1u);
        _grid_len = device.create_buffer<float>(1u);
        _grid_min = device.create_buffer<float>(3u);
        _grid_max = device.create_buffer<float>(3u);
        tot_test = device.create_buffer<uint>(1u);

        _spectrum = spectrum;
        _dimension = 3;
        _size = viewpointcount;
        this->radius = radius;
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
        return _beta3->read(index);
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
    auto grad_pos(Expr<uint> index) const noexcept {
        return _grad_pos->read(index);
    }
    UInt push(Expr<float3> position, Expr<float3> beta, Expr<float3> wi, Expr<uint> pixel_id, Expr<uint> surface_tag) {
        auto index = _tot->atomic(0u).fetch_add(1u);
        _wo->write(index, wi);
        _position->write(index, position);
        _pixel_id->write(index, pixel_id);
        _surface_tags->write(index, surface_tag);
        _beta3->write(index, beta);
        for (auto i = 0u; i < 3u; ++i)
            _grid_min->atomic(i).fetch_min(position[i]);
        for (auto i = 0u; i < 3u; ++i)
            _grid_max->atomic(i).fetch_max(position[i]);
        _nxt->write(index, 0u);
        return index;
    }
    void set_grad_pos(Expr<uint> index, Expr<float3> grad) {
        _grad_pos->write(index, grad);
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
class PixelIndirect {
public:
    Buffer<float> _radius;
    Buffer<uint> _cur_n;
    Buffer<uint> _n_photon;
    Buffer<float> _phi;
    Buffer<float> _tau;
    Buffer<float> _weight;
    const Spectrum::Instance *_spectrum;
    bool _shared_radius;
    uint _viewpoint_per_iter;
    float _clamp;

public:
    PixelIndirect(uint viewpoint_per_iter, const Spectrum::Instance *spectrum) {
        _spectrum = spectrum;
        _clamp = 1024.0;
        auto device = spectrum->pipeline().device();
        auto dimension = 3u;//always save rgb
        _shared_radius = true;
        if (_shared_radius) {
            _radius = device.create_buffer<float>(1);
            _cur_n = device.create_buffer<uint>(1);
            _n_photon = device.create_buffer<uint>(1);
        } else {
            _radius = device.create_buffer<float>(viewpoint_per_iter);
            _cur_n = device.create_buffer<uint>(viewpoint_per_iter);
            _n_photon = device.create_buffer<uint>(viewpoint_per_iter);
        }
        _phi = device.create_buffer<float>(viewpoint_per_iter * dimension);
        _tau = device.create_buffer<float>(viewpoint_per_iter * dimension);
        _weight = device.create_buffer<float>(viewpoint_per_iter);
        _viewpoint_per_iter = viewpoint_per_iter;
    }
    void write_radius(Expr<uint> pixel_id, Expr<float> value) noexcept {
        if (!_shared_radius) {
            _radius->write(pixel_id, value);
        } else {
            _radius->write(0u, value);
        }
    }
    void write_cur_n(Expr<uint> pixel_id, Expr<uint> value) noexcept {
        if (!_shared_radius) {
            _cur_n->write(pixel_id, value);
        } else {
            _cur_n->write(0u, value);
        }
    }

    void write_n_photon(Expr<uint> pixel_id, Expr<uint> value) noexcept {
        if (!_shared_radius) {
            _n_photon->write(pixel_id, value);
        } else {
            _n_photon->write(0u, value);
        }
    }

    void reset_phi(Expr<uint> pixel_id) noexcept {
        auto dimension = 3u;
        for (auto i = 0u; i < dimension; ++i)
            _phi->write(pixel_id * dimension + i, 0.f);
    }

    void reset_tau(Expr<uint> pixel_id) noexcept {
        auto dimension = 3u;
        for (auto i = 0u; i < dimension; ++i)
            _tau->write(pixel_id * dimension + i, 0.f);
    }

    auto radius(Expr<uint> pixel_id) const noexcept {
        if (!_shared_radius) {
            return _radius->read(pixel_id);
        } else {
            return _radius->read(0u);
        }
    }

    //tau=(tau+clamp(phi))*value, see pixel_info_update for useage
    void update_tau(Expr<uint> pixel_id, Expr<float> value) noexcept {
        auto dimension = 3u;
        auto thershold = _clamp;
        for (auto i = 0u; i < dimension; ++i) {
            auto old_tau = _tau->read(pixel_id * dimension + i);
            auto phi = _phi->read(pixel_id * dimension + i);
            phi = max(-thershold, min(phi, thershold));//-thershold for wavelength sampling
            _tau->write(pixel_id * dimension + i, (old_tau + phi) * value);
        }
    }

    auto n_photon(Expr<uint> pixel_id) const noexcept {
        if (!_shared_radius) {
            return _n_photon->read(pixel_id);
        } else {
            return _n_photon->read(0u);
        }
    }

    auto cur_n(Expr<uint> pixel_id) const noexcept {
        if (!_shared_radius) {
            return _cur_n->read(pixel_id);
        } else {
            return _cur_n->read(0u);
        }
    }
    

    auto phi(Expr<uint> pixel_id) const noexcept {
        auto dimension = 3u;
        Float3 ret;
        for (auto i = 0u; i < dimension; ++i)
            ret[i] = _phi->read(pixel_id * dimension + i);
        return ret;
    }
    
    auto tau(Expr<uint> pixel_id) const noexcept {
        auto dimension = 3u;
        Float3 ret;
        for (auto i = 0u; i < dimension; ++i)
            ret[i] = _tau->read(pixel_id * dimension + i);
        return ret;
    }

    void add_cur_n(Expr<uint> pixel_id, Expr<uint> value) noexcept {
        if (!_shared_radius) {
            _cur_n->atomic(pixel_id).fetch_add(value);
        } else {
            _cur_n->atomic(0u).fetch_add(value);
        }
    }

    void add_phi(Expr<uint> pixel_id, Expr<float3> phi) noexcept {
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

unique_ptr<PixelIndirect> indirect;
unique_ptr<ViewPointMap> viewpoint_map;

PYBIND11_MODULE(_lrapi, m) {
    m.doc() = "LuisaRender API";// optional module docstring
    // log
    m.def("log_info_test", [](const char *msg) { LUISA_INFO("{}", msg); });
    // util function for uniform encoding
    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");
    m.def("init", []() {
        log_level_info();
        LUISA_INFO("LuisaRender API init");
    });
    m.def("init_viewpointmap", [](uint viewpoints, float radius) {
        LUISA_INFO("LuisaRender init_viewpointmap");
        viewpoint_map = make_unique<ViewPointMap>(viewpoints, scene_python._pipeline->spectrum(), radius);
        indirect = make_unique<PixelIndirect>(viewpoints, scene_python._pipeline->spectrum());

        Kernel1D viewpointreset_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            viewpoint_map->reset(index);
        };

        Kernel1D indirect_initialize_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            indirect->write_radius(index, radius);
            viewpoint_map->write_grid_len(radius);
            indirect->write_cur_n(index, 0u);
            indirect->write_n_photon(index, 0u);
            indirect->reset_phi(index);
            indirect->reset_tau(index);
        };
        auto viewpointreset = scene_python._pipeline->device().compile(viewpointreset_kernel);
        auto indirect_initialize = scene_python._pipeline->device().compile(indirect_initialize_kernel);
        *scene_python._stream << viewpointreset().dispatch(viewpoints);
        *scene_python._stream << indirect_initialize().dispatch(viewpoints);
        *scene_python._stream << synchronize();

    });
    m.def("add_viewpoint", [](uint64_t pos_ptr, uint64_t wi_ptr, uint64_t beta_ptr, uint size, bool debug) {

        auto pos_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(pos_ptr),size*3);
        auto wi_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(wi_ptr),size*3);
        auto beta_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(beta_ptr),size*3);

        Kernel1D viewpointreset_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            viewpoint_map->reset(index);
        };
        Kernel1D add_viewpoint_kernel = [&](Bool debug) noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            auto pos = make_float3(pos_buffer->read(index*3+0),pos_buffer->read(index*3+1),pos_buffer->read(index*3+2));
            auto wi = make_float3(wi_buffer->read(index*3+0),wi_buffer->read(index*3+1),wi_buffer->read(index*3+2));
            auto beta = make_float3(beta_buffer->read(index*3+0),beta_buffer->read(index*3+1),beta_buffer->read(index*3+2));
            $if(pos[0]>-500.f)
            {
                $if(debug)
                {
                    device_log("index {} pos: {}, wi: {}, beta: {}",index, pos, wi, beta);
                };
                viewpoint_map->push(pos, beta, wi, index, 0u);
            };
            //device_log("index {} pos: {}, wi: {}, beta: {}",index, pos, wi, beta);
        };

        Kernel1D build_grid_kernel = [&]() noexcept {
            auto index = static_cast<UInt>(dispatch_x());
            $if(viewpoint_map->nxt(index) == 0u) {
                viewpoint_map->link(index);
            };
        };

        auto viewpointreset = scene_python._pipeline->device().compile(viewpointreset_kernel);
        *scene_python._stream << viewpointreset().dispatch(viewpoint_map->size()) << synchronize();

        auto add_viewpoint = scene_python._device->compile(add_viewpoint_kernel);
        auto build_grid = scene_python._device->compile(build_grid_kernel);

        *scene_python._stream << add_viewpoint(debug).dispatch(size) << synchronize();
        LUISA_INFO("LuisaRender add_viewpoint");
        
        *scene_python._stream << build_grid().dispatch(size) << synchronize();
        LUISA_INFO("LuisaRender build_grid");
    });

    m.def("indirect_update", []() {
        LUISA_INFO("LuisaRender indirect_update");
        Kernel1D indirect_update_kernel = [&]() noexcept {
            set_block_size(16u, 16u, 1u);
            auto pixel_id = dispatch_x();
            indirect->pixel_info_update(pixel_id);
        };

        Kernel1D shared_update_kernel = [&]() noexcept {
            indirect->shared_update();
            viewpoint_map->write_grid_len(indirect->radius(0u));
        };

        auto indirect_update = scene_python._device->compile(indirect_update_kernel);
        auto shared_update = scene_python._device->compile(shared_update_kernel);

        *scene_python._stream << indirect_update().dispatch(viewpoint_map->size());
        *scene_python._stream << shared_update().dispatch(1u) << synchronize();
    });

    m.def("get_indirect", [](uint64_t buffer_ptr, uint size, uint tot_photon, bool debug) {
        LUISA_INFO("LuisaRender get_indirect");
        auto ind_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(buffer_ptr),size*3);
        Kernel1D indirect_draw_kernel = [&](Bool debug) noexcept {
            auto index = dispatch_x();
            auto r = indirect->radius(index);
            auto tau = indirect->tau(index);
            auto L = tau / (tot_photon * pi * r * r);
            $if(debug)
            {
                device_log("index {} tau: {}, L: {}, r: {}",index, tau, L, r);
            };
            ind_buffer->write(index*3+0, L[0]);
            ind_buffer->write(index*3+1, L[1]);
            ind_buffer->write(index*3+2, L[2]);
            //camera->film()->accumulate(pixel_id, L, 0.5f * spp);
        };
        auto indirect_draw = scene_python._device->compile(indirect_draw_kernel);
        *scene_python._stream << indirect_draw(debug).dispatch(size) << synchronize();
        //return reinterpret_cast<uint64_t>(ind_buffer.native_handle());
    });

    m.def("update_grad", [](uint64_t grad_buffer_ptr, uint size, bool debug) {
        LUISA_INFO("LuisaRender update_grad");
        auto grad_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(grad_buffer_ptr),size*3);
        Kernel1D update_grad_kernel = [&](Bool debug) noexcept {
            auto index = dispatch_x();
            auto grad = make_float3(grad_buffer->read(index*3+0),grad_buffer->read(index*3+1),grad_buffer->read(index*3+2));
            $if(debug){
                device_log("index {} grad: {}",index, grad);
            };
            viewpoint_map->set_grad_pos(index, grad);
        };
        auto update_grad = scene_python._device->compile(update_grad_kernel);
        *scene_python._stream << update_grad(debug).dispatch(size) << synchronize();
    });

    m.def("accum_ind", [](uint64_t pos_ptr,uint64_t wi_ptr, uint64_t beta_ptr, uint size, bool debug) {

        LUISA_INFO("LuisaRender accum ind");
        auto pos_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(pos_ptr),size*3);
        auto wi_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(wi_ptr),size*3);
        auto beta_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(beta_ptr),size*3);

        Kernel1D query_kernel = [&](Bool debug) noexcept{
            auto index = dispatch_x();
            
            auto pos = make_float3(pos_buffer->read(index*3+0),pos_buffer->read(index*3+1),pos_buffer->read(index*3+2));
            auto wi_local = make_float3(wi_buffer->read(index*3+0),wi_buffer->read(index*3+1),wi_buffer->read(index*3+2));
            auto beta = make_float3(beta_buffer->read(index*3+0),beta_buffer->read(index*3+1),beta_buffer->read(index*3+2));
            
            auto grid = viewpoint_map->point_to_grid(pos);
            $if(debug){
                device_log("index {} pos: {}, wi: {}, beta: {} grid {}",index, pos, wi_local, beta, grid);
            };
            auto radius = viewpoint_map->radius;
            Float3 ind{0.0f, 0.0f, 0.0f};
            $for(x, grid.x - 1, grid.x + 2) {
                $for(y, grid.y - 1, grid.y + 2) {
                    $for(z, grid.z - 1, grid.z + 2) {
                        Int3 check_grid{x, y, z};
                        auto viewpoint_index = viewpoint_map->grid_head(viewpoint_map->grid_to_index(check_grid));
                        $if(debug){
                            device_log("index {} viewpoint_index: ",index, viewpoint_index);
                        };
                        $while(viewpoint_index != ~0u) {
                            auto position = viewpoint_map->position(viewpoint_index);
                            auto pixel_id = viewpoint_map->pixel_id(viewpoint_index);
                            auto dis = distance(position, pos);
                            $if(dis <= radius) {
                                auto viewpointwo = viewpoint_map->wo(viewpoint_index);
                                auto viewpointbeta = viewpoint_map->beta(viewpoint_index);
                                auto rel_dis = dis / radius;
                                auto weight = 3.5f*(1.0f- 6 * pow(rel_dis, 5.) + 15 * pow(rel_dis, 4.) - 10 * pow(rel_dis, 3.));
                                auto Phi = viewpointbeta*beta*weight/max(0.01f,abs_cos_theta(wi_local));
                                indirect->add_phi(pixel_id, Phi);
                                indirect->add_cur_n(pixel_id, 1u);
                                $if(debug){
                                    device_log("index {} pos: {}, wi: {}, beta: {}, weight: {}, Phi: {}, pixel_id {}",index, pos, wi_local, beta, weight, Phi, pixel_id);
                                };
                            };
                            viewpoint_index = viewpoint_map->nxt(viewpoint_index);
                        };
                    };
                };
            };
            // pos_buffer->write(index*3+0, ind[0]/(3.1415926f*radius*radius));
            // pos_buffer->write(index*3+1, ind[1]/(3.1415926f*radius*radius));
            // pos_buffer->write(index*3+2, ind[2]/(3.1415926f*radius*radius));
        };
        auto query_ind = scene_python._device->compile(query_kernel);
        *scene_python._stream << query_ind(debug).dispatch(size) << synchronize();
        // return reinterpret_cast<uint64_t>(pos_buffer.native_handle());
    });

    m.def("compute_ind_grad", [](uint64_t pos_ptr,uint64_t wi_ptr, uint64_t beta_ptr, uint size, uint tot_photon) {

        LUISA_INFO("LuisaRender compute_ind_grad");
        auto pos_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(pos_ptr),size*3);
        auto wi_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(wi_ptr),size*3);
        auto beta_buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(beta_ptr),size*3);

        Kernel1D compute_kernel = [&]() noexcept{
            auto index = dispatch_x();
        
            auto pos = make_float3(pos_buffer->read(index*3+0),pos_buffer->read(index*3+1),pos_buffer->read(index*3+2));
            auto wi_local = make_float3(wi_buffer->read(index*3+0),wi_buffer->read(index*3+1),wi_buffer->read(index*3+2));
            auto beta = make_float3(beta_buffer->read(index*3+0),beta_buffer->read(index*3+1),beta_buffer->read(index*3+2));
    
            auto grid = viewpoint_map->point_to_grid(pos);
            Float3 grad_pos{0.0f, 0.0f, 0.0f};
            Float3 grad_beta{0.0f, 0.0f, 0.0f};
            Float grad_dis{0.0f};
            $for(x, grid.x - 1, grid.x + 2) {
                $for(y, grid.y - 1, grid.y + 2) {
                    $for(z, grid.z - 1, grid.z + 2) {
                        Int3 check_grid{x, y, z};
                        auto viewpoint_index = viewpoint_map->grid_head(viewpoint_map->grid_to_index(check_grid));
                        $while(viewpoint_index != ~0u) {
                            auto position = viewpoint_map->position(viewpoint_index);
                            auto pixel_id = viewpoint_map->pixel_id(viewpoint_index);
                            auto radius = indirect->radius(pixel_id);
                            auto dis = distance(position, pos);
                            $if(dis <= radius) {
                                auto viewpointwo = viewpoint_map->wo(viewpoint_index);
                                auto viewpointbeta = viewpoint_map->beta(viewpoint_index);
                                auto grad_view = viewpoint_map->grad_pos(pixel_id);
                                auto Phi = viewpointbeta/max(0.01f,abs_cos_theta(wi_local));
                                $autodiff {
                                    requires_grad(pos, beta);
                                    auto rel_dis = distance(position, pos) / radius;
                                    requires_grad(rel_dis);
                                    auto rel3 = rel_dis*rel_dis*rel_dis;
                                    auto weight = 3.5f*(1- 6*rel3*rel_dis*rel_dis + 15*rel3*rel_dis - 10*rel3);
                                    auto Phi_beta_weight = Phi * beta * weight;
                                    auto contrib = Phi_beta_weight / (tot_photon * pi * radius * radius);
                                    auto dldPhi = (contrib[0u]*grad_view[0] + contrib[1u]*grad_view[1] + contrib[2u]*grad_view[2]);
                                    backward(dldPhi);
                                    grad_pos += grad(pos);
                                    grad_beta += grad(beta);
                                    grad_dis = grad(rel_dis);
                                };
                            };
                            viewpoint_index = viewpoint_map->nxt(viewpoint_index);
                        };
                    };
                };
            };
            pos_buffer->write(index*3+0, grad_pos[0]);
            pos_buffer->write(index*3+0, grad_pos[1]);
            pos_buffer->write(index*3+0, grad_pos[2]);
            beta_buffer->write(index*3+0, grad_beta[0]);
            beta_buffer->write(index*3+0, grad_beta[1]);
            beta_buffer->write(index*3+0, grad_beta[2]);
        };
        auto compute_grad = scene_python._device->compile(compute_kernel);
        *scene_python._stream << compute_grad().dispatch(size) << synchronize();
    });

    m.def("load_scene", [](std::vector<std::string> &argvs){
        int argc = argvs.size();
        LUISA_INFO("Argc: {}", argc);
        vector<char*> pointerVec(argc);
        for(unsigned i = 0; i < argc; ++i)
        {
            LUISA_INFO("Argv: {} {}", i, argvs[i]);
            pointerVec[i] = argvs[i].data();
        } 
        char** argv = pointerVec.data();
        luisa::compute::Context context{argv[0]};
        auto macros = parse_cli_macros(argc, argv);
        for (auto &&[k, v] : macros) {
            LUISA_INFO("Found CLI Macro: {} = {}", k, v);
        }
        auto options = parse_cli_options(argc, argv);  
        auto backend = options["backend"].as<luisa::string>();
        auto index = options["device"].as<int32_t>();
        auto path = options["scene"].as<std::filesystem::path>();
        
        compute::DeviceConfig config;
        config.device_index = index;
        config.inqueue_buffer_limit = false; // Do not limit the number of in-queue buffers --- we are doing offline rendering!
        auto device = context.create_device(backend, &config);
        scene_python._device = luisa::make_unique<Device>(std::move(device));
        auto scene_desc = SceneParser::parse(path, macros);
        
        auto stream = scene_python._device->create_stream(StreamTag::GRAPHICS);
        scene_python._scene = Scene::create(context, scene_desc.get());
        scene_python._stream = luisa::make_unique<Stream>(std::move(stream));
        scene_python._pipeline = Pipeline::create(*scene_python._device, *scene_python._stream, *scene_python._scene);
    });

    m.def("render", []() {
        //LUISA_INFO("LuisaRender API render_scene");
        auto res = scene_python._pipeline->render_with_return(*scene_python._stream);
        scene_python._stream->synchronize();
        std::vector<uint64_t> res_vec(res.size());
        for (int i = 0; i < res.size(); i++) {
            res_vec[i] = reinterpret_cast<uint64_t>(res[i]);
        }
        //LUISA_INFO("res_vec: {}",res_vec[0]);
        return res_vec;
    });

    // m.def("update_texture", [](uint tex_id, float4 texture_buffer) {
    //     LUISA_INFO("LuisaRender Update Texture");
    //     scene_python._pipeline->update_texture(*scene_python._stream, tex_id, texture_buffer);
    // });


    m.def("update_scene", [](std::vector<ParamStruct> params) {
        //LUISA_INFO("LuisaRender Update Scene");
        luisa::vector<float4> constants{};
        luisa::vector<Buffer<float4>> textures{};
        luisa::vector<Buffer<float>> geoms{};
        luisa::vector<uint> constants_id{};
        luisa::vector<uint> textures_id{};
        luisa::vector<uint> geoms_id{};
        for (auto param: params) {
            //LUISA_INFO("Param: {} {} {} {} {}", param.type, param.id, param.size, param.buffer_ptr, param.value);
            if(param.type == "constant") {
                constants_id.push_back(param.id);
                constants.push_back(param.value);
            }
            else if(param.type == "texture") {
                textures_id.push_back(param.id);
                auto buffer = scene_python._pipeline->device().import_external_buffer<float4>(reinterpret_cast<void *>(param.buffer_ptr),param.size);
                LUISA_INFO("Param buffer created");
                textures.push_back(std::move(buffer));
            }
            else if(param.type == "geom") {
                geoms_id.push_back(param.id);
                auto buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(param.buffer_ptr),param.size);
                geoms.push_back(std::move(buffer));
            }
        }
        //LUISA_INFO("geom_id_size is {}", geoms_id.size());
        scene_python._pipeline->differentiation()->update_parameter_from_external(*scene_python._stream, constants_id, constants, textures_id, textures, geoms_id, geoms);
    });

    m.def("get_scene_param", [](std::vector<ParamStruct> params) {
        //LUISA_INFO("LuisaRender API get_parameter start");
        luisa::vector<uint> constants_id{};
        luisa::vector<uint> textures_id{};
        luisa::vector<uint> geoms_id{};
        for (auto param: params) {
            if(param.type == "constant") {
                constants_id.push_back(param.id);
            }
            else if(param.type == "texture") {
                textures_id.push_back(param.id);
            }
            else if(param.type == "geom") {
                geoms_id.push_back(param.id);
            }
        }
        auto [geom_param, geom_size, tri_param, tri_size] = scene_python._pipeline->differentiation()->get_parameter_from_external(*scene_python._stream, constants_id, textures_id, geoms_id);
        // std::vector<float> ret_con_param(constants_id.size());
        // std::vector<uint64_t> ret_tex_param(textures_id.size());
        // std::vector<uint> ret_tex_size(textures_id.size());
        std::vector<uint64_t> ret_geom_param(geoms_id.size());
        std::vector<uint> ret_geom_size(geoms_id.size());
        std::vector<uint64_t> ret_tri_param(geoms_id.size());
        std::vector<uint> ret_tri_size(geoms_id.size());
        // for (int i = 0; i < ret_con_param.size(); i++) {
        //     ret_con_param[i] = constant_param[i];
        // }
        // for (int i = 0; i < ret_tex_param.size(); i++) {
        //     ret_tex_param[i] = reinterpret_cast<uint64_t>(tex_param[i]);
        //     ret_tex_size[i] = tex_size[i];
        // }
        for (int i = 0; i < ret_geom_param.size(); i++) {
            ret_geom_param[i] = reinterpret_cast<uint64_t>(geom_param[i]);
            ret_geom_size[i] = geom_size[i];
            ret_tri_param[i] = reinterpret_cast<uint64_t>(tri_param[i]);
            ret_tri_size[i] = tri_size[i];
            //LUISA_INFO("LuisaRender API get_parameter {} {} {}", i, ret_geom_size[i], ret_geom_param[i]);
        }
        //LUISA_INFO("LuisaRender API get_parameter finish");
        return std::make_tuple(ret_geom_param, ret_geom_size, ret_tri_param, ret_tri_size);
    });

    m.def("render_backward", [](std::vector<uint64_t> grad_ptr,std::vector<uint> sizes){
        
        //LUISA_INFO("LuisaRender API render_backward");
        //scene_python._pipeline->differentiation()->clear_gradients(*scene_python._stream);
        luisa::vector<Buffer<float>> grad_buffer{grad_ptr.size()};
        for (int i = 0; i < grad_ptr.size(); i++) {
            auto buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(grad_ptr[i]),sizes[i]);
            grad_buffer[i] = std::move(buffer);
        }
        auto res = scene_python._pipeline->render_diff(*scene_python._stream, grad_buffer);
        scene_python._stream->synchronize();
        std::vector<uint64_t> res_vec(res.size());
        for (int i = 0; i < res.size(); i++) {
            res_vec[i] = reinterpret_cast<uint64_t>(res[i]);
        }
        return res_vec;
    });

    m.def("get_gradients", [](){
        //luisa::vector<void*> tex_grad;
        //luisa::vector<void*> geom_grad;
        auto [tex_grad, geom_grad] = scene_python._pipeline->differentiation()->get_gradients(*scene_python._stream);
        LUISA_INFO("LuisaRender API get_gradients");
        std::vector<uint64_t> tex_res(tex_grad.size());
        std::vector<uint64_t> geom_res(geom_grad.size());
        for (int i = 0; i < tex_res.size(); i++) {
            tex_res[i] = reinterpret_cast<uint64_t>(tex_grad[i]);
        }
        for (int i = 0; i < geom_res.size(); i++) {
            geom_res[i] = reinterpret_cast<uint64_t>(geom_grad[i]);
        }
        return std::make_pair(tex_res, geom_res);
    });
    
    py::class_<ParamStruct>(m, "ParamStruct")
        .def(py::init<>())
        .def(py::init<std::string, uint, uint, uint64_t, float4>())
        .def_readwrite("type", &ParamStruct::type)
        .def_readwrite("id", &ParamStruct::id)
        .def_readwrite("size", &ParamStruct::size)
        .def_readwrite("buffer_ptr", &ParamStruct::buffer_ptr)
        .def_readwrite("value", &ParamStruct::value);
    
    // py::class_<SceneDesc>(m, "SceneDesc")
    //     .def(py::init<>())
    //     .def("nodes", &SceneDesc::nodes)
    //     .def("node", &SceneDesc::node)
    //     .def("root", &SceneDesc::root)
    //     .def("reference", &SceneDesc::reference)
    //     .def("define", &SceneDesc::define)
    //     .def("define_root", &SceneDesc::define_root)
    //     .def("register_path", &SceneDesc::register_path);
    
    // py::class_<Pipeline>(m, "Pipeline")
    //     .def(py::init<>())
    //     .def("render", &Pipeline::render);
    //     .def("update_texture", &Pipeline::update_texture);
    //     .def("update_mesh", &Pipeline::update_mesh);
    
}

