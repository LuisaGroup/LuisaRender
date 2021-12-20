//
// Created by Mike on 2021/12/7.
//

#include <span>

#include <cxxopts.hpp>
#include <luisa-compute.h>

#include <sdl/scene_desc.h>

[[nodiscard]] auto parse_cli_options(int argc, const char *const *argv) noexcept {
    cxxopts::Options cli{"megakernel_path_tracing"};
    cli.add_option("", "b", "backend", "Compute backend name", cxxopts::value<std::string>(), "<backend>");
    cli.add_option("", "d", "device", "Compute device index", cxxopts::value<uint32_t>()->default_value("0"), "<index>");
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
    if (options["backend"].count() == 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "Compute backend not specified.");
        std::cout << cli.help() << std::endl;
        exit(-1);
    }
    return options;
}

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

void dump(std::ostream &os, const SceneNodeDesc *node, size_t indent_level = 0) noexcept {
    auto indent = [&os](auto n) noexcept {
        for (auto i = 0u; i < n; i++) { os << "  "; }
    };
    os << node->impl_type() << " {";
    for (auto &&[prop, values] : node->properties()) {
        os << "\n";
        indent(indent_level + 1u);
        os << prop << " ";
        std::visit(
            [&](auto &&v) noexcept {
                using T = std::remove_cvref_t<decltype(v)>;
                if constexpr (std::is_same_v<T, SceneNodeDesc::string_list>) {
                    os << "{";
                    if (!v.empty()) {
                        os << " \"" << v.front() << '"';
                        for (auto i = 1u; i < v.size(); i++) {
                            os << ", \"" << v[i] << '"';
                        }
                        os << " ";
                    }
                    os << "}";
                } else if constexpr (std::is_same_v<T, SceneNodeDesc::node_list>) {
                    if (v.size() == 1u && v.front()->is_internal()) {
                        os << ": ";
                        dump(os, v.front(), indent_level + 1u);
                    } else {
                        os << "{";
                        if (!v.empty()) {
                            os << " @" << v.front()->identifier();
                            for (auto i = 1u; i < v.size(); i++) {
                                os << ", @" << v[i]->identifier();
                            }
                            os << " ";
                        }
                        os << "}";
                    }
                } else {
                    os << "{";
                    if (!v.empty()) {
                        os << " " << v.front();
                        for (auto i = 1u; i < v.size(); i++) {
                            os << ", " << v[i];
                        }
                        os << " ";
                    }
                    os << "}";
                }
            },
            values);
    }
    if (!node->properties().empty()) {
        os << "\n";
        indent(indent_level);
    }
    os << "}";
}

void dump(std::ostream &os, const SceneDesc &scene) noexcept {
    auto flags = os.flags();
    os << std::boolalpha;
    for (auto &&node : scene.nodes()) {
        os << scene_node_tag_description(node->tag()) << " "
           << node->identifier() << " : ";
        dump(os, node.get());
        os << "\n\n";
    }
    os << "// entry\n";
    dump(os, scene.root());
    os << std::endl;
    os.flags(flags);
}

class Base {
public:
    virtual ~Base() noexcept = default;
    [[nodiscard]] virtual Float foo(Int x, Float y) const noexcept = 0;
};

[[nodiscard]] Float use_base(const Base &base, Int x, Float y) noexcept {
    return base.foo(x, y);
}

class DerivedA : public Base {
public:
    [[nodiscard]] Float foo(Int x, Float y) const noexcept override {
        return cast<float>(x) + y;
    }
};

class DerivedB : public Base {
private:
    Buffer<float> _buffer;
public:
    [[nodiscard]] Float foo(Int x, Float y) const noexcept override {
        return _buffer[x] + y;
    }
};

int main(int argc, char *argv[]) {

    luisa::compute::Context context{argv[0]};

    auto options = parse_cli_options(argc, argv);
    auto backend = options["backend"].as<std::string>();
    auto index = options["device"].as<uint32_t>();

    auto device = context.create_device(backend, {{"index", index}});

    SceneDesc scene;
    auto camera = scene.define("camera", SceneNodeTag::CAMERA, "ThinLens");
    auto film = camera->define_internal("film", "RGB");
    film->add_property("resolution", SceneNodeDesc::number_list{1.0, 1.0});
    film->add_property("filter", scene.reference("filter"));
    auto filter = scene.define("filter", SceneNodeTag::FILTER, "Gaussian");
    filter->add_property("radius", 1.5);
    auto integrator = scene.define("integrator", SceneNodeTag::INTEGRATOR, "Path");
    integrator->add_property("sampler", scene.reference("sampler"));
    auto sampler = scene.define("sampler", SceneNodeTag::SAMPLER, "Independent");
    sampler->add_property("spp", 1024.0);
    auto root = scene.define_root();
    root->add_property("integrator", integrator);
    root->add_property("camera", camera);

    std::ostringstream os;
    dump(os, scene);
    LUISA_INFO("Scene dump:\n{}", os.str());
}
