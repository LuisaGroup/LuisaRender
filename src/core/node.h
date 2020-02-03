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

#include <util/noncopyable.h>

#include "data_types.h"
#include "device.h"

namespace luisa {

class ParameterSet;

template<typename BaseClass>
using NodeCreator = std::function<
std::unique_ptr<BaseClass>(Device
*, const ParameterSet &)>;

template<typename BaseClass>
class NodeCreatorRegistry {

private:
    std::unordered_map<std::string_view, NodeCreator<BaseClass>> _creators;

public:
    NodeCreatorRegistry() noexcept = default;
    void emplace(std::string_view derived_name, NodeCreator<BaseClass> creator) noexcept {
        assert(_creators.find(derived_name) == _creators.end());
        _creators.emplace(derived_name, std::move(creator));
    }
    NodeCreator<BaseClass> &operator[](std::string_view derived_name) noexcept {
        auto iter = _creators.find(derived_name);
        assert(iter != _creators.end());
        return iter->second;
    }
};

#define MAKE_BASE_NODE_CLASS_MATCHER(BaseClass)  \
    class BaseClass;  \
    namespace _impl { template<typename T, std::enable_if_t<std::is_base_of_v<BaseClass, T>, int> = 0> auto base_node_class_impl(T &&) -> BaseClass * {} }

MAKE_BASE_NODE_CLASS_MATCHER(Filter)
MAKE_BASE_NODE_CLASS_MATCHER(Film)
MAKE_BASE_NODE_CLASS_MATCHER(Camera)
MAKE_BASE_NODE_CLASS_MATCHER(Geometry)
MAKE_BASE_NODE_CLASS_MATCHER(Light)
MAKE_BASE_NODE_CLASS_MATCHER(Material)
MAKE_BASE_NODE_CLASS_MATCHER(Integrator)

#undef MAKE_BASE_NODE_CLASS_MATCHER

namespace _impl {

template<typename BaseClass>
void register_node_creator(std::string_view derived_name, NodeCreator<BaseClass> creator) noexcept {
    BaseClass::creators.emplace(derived_name, std::move(creator));
}

template<typename T>
using BaseNodeClass = std::remove_pointer_t<decltype(base_node_class_impl(std::declval<T>()))>;

}

#define LUISA_MAKE_NODE_CREATOR_REGISTRY(BaseClass)                                                                               \
    friend class ParameterSet;                                                                                                    \
    friend void _impl::register_node_creator<BaseClass>(std::string_view derived_name, NodeCreator<BaseClass> creator) noexcept;  \
    inline static NodeCreatorRegistry<BaseClass> creators

#define LUISA_REGISTER_NODE_CREATOR(derived_type_name, DerivedClass)                                                                    \
    namespace _impl {                                                                                                                   \
        static struct Register##DerivedClass##Helper {                                                                                  \
            Register##DerivedClass##Helper() noexcept {                                                                                 \
                register_node_creator<BaseNodeClass<DerivedClass>>(derived_type_name, [](Device *device, const ParameterSet &params) {  \
                    return std::make_unique<DerivedClass>(device, params);                                                              \
                });                                                                                                                     \
            }                                                                                                                           \
        } _registration_##DerivedClass{};                                                                                               \
    }

class Node : public Noncopyable {

public:
    friend class ParameterSet;

private:
    Device *_device;

public:
    explicit Node(Device *device) noexcept : _device{device} {}
    virtual ~Node() noexcept = default;
    [[nodiscard]] Device &device() noexcept { return *_device; }
    
};

}
