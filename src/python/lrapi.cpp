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
        LUISA_INFO("LuisaRender API render_scene");
        auto res = scene_python._pipeline->render_with_return(*scene_python._stream);
        scene_python._stream->synchronize();
        std::vector<uint64_t> res_vec(res.size());
        for (int i = 0; i < res.size(); i++) {
            res_vec[i] = reinterpret_cast<uint64_t>(res[i]);
        }
        LUISA_INFO("res_vec: {}",res_vec[0]);
        return res_vec;
    });

    // m.def("update_texture", [](uint tex_id, float4 texture_buffer) {
    //     LUISA_INFO("LuisaRender Update Texture");
    //     scene_python._pipeline->update_texture(*scene_python._stream, tex_id, texture_buffer);
    // });


    m.def("update_scene", [](std::vector<ParamStruct> params) {
        LUISA_INFO("LuisaRender Update Scene");
        luisa::vector<float4> constants{};
        luisa::vector<Buffer<float4>> textures{};
        luisa::vector<Buffer<float>> geoms{};
        luisa::vector<uint> constants_id{};
        luisa::vector<uint> textures_id{};
        luisa::vector<uint> geoms_id{};
        for (auto param: params) {
            LUISA_INFO("Param: {} {} {} {} {}", param.type, param.id, param.size, param.buffer_ptr, param.value);
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
        scene_python._pipeline->differentiation()->update_parameter_from_external(*scene_python._stream, constants_id, constants, textures_id, textures, geoms_id, geoms);
    });

    m.def("render_backward", [](std::vector<uint64_t> grad_ptr,std::vector<uint> sizes){
        LUISA_INFO("LuisaRender API render_backward");
        //scene_python._pipeline->differentiation()->clear_gradients(*scene_python._stream);
        luisa::vector<Buffer<float>> grad_buffer{grad_ptr.size()};
        for (int i = 0; i < grad_ptr.size(); i++) {
            auto buffer = scene_python._pipeline->device().import_external_buffer<float>(reinterpret_cast<void *>(grad_ptr[i]),sizes[i]);
            grad_buffer[i] = std::move(buffer);
        }
        scene_python._pipeline->render_diff(*scene_python._stream, grad_buffer);
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

