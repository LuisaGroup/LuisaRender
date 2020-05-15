//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <variant>
#include <string>
#include <unordered_map>
#include <type_traits>
#include <functional>
#include <memory>
#include <iostream>

#include <util/concepts.h>
#include <util/logging.h>
#include "data_types.h"
#include "device.h"

namespace luisa {
class ParameterSet;
}

namespace luisa {

#define LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(BaseClass)                          \
    class BaseClass;                                                                      \
    namespace detail {                                                                    \
        template<typename T, std::enable_if_t<std::is_base_of_v<BaseClass, T>, int> = 0>  \
        auto plugin_base_class_match(T *) { return static_cast<BaseClass *>(nullptr); }   \
                                                                                          \
        template<typename T, std::enable_if_t<std::is_same_v<BaseClass, T>, int> = 0>     \
        constexpr std::string_view plguin_base_class_name(T *) {                          \
            using namespace std::string_view_literals;                                    \
            return #BaseClass""sv;                                                        \
        }                                                                                 \
    }

LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(Filter)
LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(Film)
LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(Camera)
LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(Shape)
LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(Transform)
LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(Light)
LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(Material)
LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(Integrator)
LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(Render)
LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME(Sampler)

#undef LUISA_MAKE_PLUGIN_BASE_CLASS_MATCHER_AND_NAME

template<typename T>
using PluginBaseClass = std::remove_pointer_t<decltype(detail::plugin_base_class_match(static_cast<T *>(nullptr)))>;

template<typename T>
constexpr auto plugin_base_class_name() noexcept { return detail::plguin_base_class_name(static_cast<PluginBaseClass<T> *>(nullptr)); }

class Plugin : Noncopyable {

protected:
    Device *_device;

public:
    explicit Plugin(Device *device) noexcept: _device{device} {}
    virtual ~Plugin() noexcept = default;
    [[nodiscard]] Device &device() { return *_device; }
    
    template<typename T>
    [[nodiscard]] static std::unique_ptr<T> create(
        Device *device,
        std::string_view derived_name_pascal_case,
        const ParameterSet &params) {
        
        auto base_name = pascal_to_snake_case(plugin_base_class_name<T>());
        auto derived_name = pascal_to_snake_case(derived_name_pascal_case);
        auto plugin_dir = device->context().runtime_path("lib") / base_name.append("s");
        using PluginCreator = T *(Device *, const ParameterSet &);
        auto creator = device->context().load_dynamic_function<PluginCreator>(plugin_dir, derived_name, "create");
        return std::unique_ptr<T>{creator(device, params)};
    }
};

}

#define LUISA_EXPORT_PLUGIN_CREATOR(PluginClass)                                                                                        \
    LUISA_DLL_EXPORT ::luisa::Plugin *create(::luisa::Device *device, const ::luisa::ParameterSet &params) {                            \
        luisa::LUISA_INFO("Creating instance of class ", #PluginClass, ", catalog: ", ::luisa::plugin_base_class_name<PluginClass>());  \
        return new PluginClass{device, params};                                                                                         \
    }
