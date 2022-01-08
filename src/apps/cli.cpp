//
// Created by Mike on 2021/12/7.
//

#include <span>

#include <cxxopts.hpp>
#include <luisa-compute.h>

#include <sdl/scene_desc.h>
#include <sdl/scene_parser.h>
#include <scene/scene.h>
#include <scene/pipeline.h>

[[nodiscard]] auto parse_cli_options(int argc, const char *const *argv) noexcept {
    cxxopts::Options cli{"megakernel_path_tracing"};
    cli.add_option("", "b", "backend", "Compute backend name", cxxopts::value<luisa::string>(), "<backend>");
    cli.add_option("", "d", "device", "Compute device index", cxxopts::value<uint32_t>()->default_value("0"), "<index>");
    cli.add_option("", "", "scene", "Path to scene description file", cxxopts::value<std::filesystem::path>(), "<file>");
    cli.allow_unrecognised_options();
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
    if (options["scene"].count() == 0u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION("Scene file not specified.");
        std::cout << cli.help() << std::endl;
        exit(-1);
    }
    if (auto unknown = options.unmatched(); !unknown.empty()) [[unlikely]] {
        luisa::string opts;
        for (auto &&u : unknown) {
            opts.append(" ").append(u);
        }
        LUISA_WARNING_WITH_LOCATION(
            "Unrecognized options: {}", opts);
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
        luisa::visit(
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
                    if (v.size() == 1u) {
                        if (v.front()->is_internal()) {
                            os << ": ";
                            dump(os, v.front(), indent_level + 1u);
                        } else {
                            os << "{ @" << v.front()->identifier() << " }";
                        }
                    } else {
                        os << "{";
                        if (!v.empty()) {
                            os << "\n";
                            indent(indent_level + 2u);
                            if (v.front()->is_internal()) {
                                dump(os, v.front(), indent_level + 2u);
                            } else {
                                os << "@" << v.front()->identifier();
                            }
                            for (auto i = 1u; i < v.size(); i++) {
                                os << ",\n";
                                indent(indent_level + 2u);
                                if (v[i]->is_internal()) {
                                    dump(os, v[i], indent_level + 2u);
                                } else {
                                    os << "@" << v[i]->identifier();
                                }
                            }
                            os << "\n";
                        }
                        indent(indent_level + 1u);
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
        return _buffer.read(x) + y;
    }
};

int main(int argc, char *argv[]) {

    luisa::compute::Context context{argv[0]};

    auto options = parse_cli_options(argc, argv);
    auto backend = options["backend"].as<luisa::string>();
    auto index = options["device"].as<uint32_t>();
    auto path = options["scene"].as<std::filesystem::path>();

    auto device = context.create_device(backend, {{"index", index}});
    Clock clock;
    auto scene_desc = SceneParser::parse(path);
    LUISA_INFO("Parse time: {} ms.", clock.toc());

    std::ostringstream os;
    dump(os, *scene_desc);
    LUISA_INFO("Scene dump:\n{}", os.str());

    auto scene = Scene::create(context, scene_desc.get());

    auto stream = device.create_stream();
    auto pipeline = Pipeline::create(device, stream, *scene);
    pipeline->render(stream);
    stream.synchronize();
}
