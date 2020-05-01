//
// Created by Mike Smith on 2020/1/31.
//

#pragma once

#include "data_types.h"
#include "mathematics.h"

namespace luisa {

struct Ray {
    packed_float3 origin;
    float min_distance;
    packed_float3 direction;
    float max_distance;
};

LUISA_DEVICE_CALLABLE inline auto make_ray(float3 o, float3 d, float t_min = 1e-4f, float t_max = INFINITY) noexcept {
    return Ray{make_packed_float3(o), t_min, make_packed_float3(d), t_max};
}

// Adapted from Ray Tracing Gems
LUISA_DEVICE_CALLABLE inline float3 offset_ray_origin(float3 p, float3 n) noexcept {
    
    constexpr auto origin = 1.0f / 32.0f;
    constexpr auto float_scale = 1.0f / 65536.0f;
    constexpr auto int_scale = 256.0f;
    
    auto of_i = make_int3(static_cast<int>(int_scale * n.x), static_cast<int>(int_scale * n.y), static_cast<int>(int_scale * n.z));
    
    auto p_i = make_float3(
        as<float>(as<int>(p.x) + (p.x < 0 ? -of_i.x : of_i.x)),
        as<float>(as<int>(p.y) + (p.y < 0 ? -of_i.y : of_i.y)),
        as<float>(as<int>(p.z) + (p.z < 0 ? -of_i.z : of_i.z)));
    
    return make_float3(
        math::abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
        math::abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
        math::abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <string>
#include <memory>
#include <string_view>
#include <map>

#include <util/exception.h>

#include "device.h"
#include "buffer.h"

namespace luisa {

class RayAttributeBufferSet {

private:
    Device *_device;
    size_t _capacity;
    std::map<std::string, std::unique_ptr<TypelessBuffer>, std::less<>> _buffers;

public:
    RayAttributeBufferSet(Device *device, size_t capacity) noexcept : _device{device}, _capacity{capacity} {}
    
    template<typename T>
    void add(std::string name) {
        LUISA_ERROR_IF_NOT(_buffers.find(name) == _buffers.end(), "ray attribute already exists: ", name);
        _buffers.emplace(name, _device->allocate_buffer(sizeof(T) * _capacity, BufferStorage::DEVICE_PRIVATE));
    }
    
    template<typename T>
    BufferView<T> view(std::string_view name) {
        auto iter = _buffers.find(name);
        LUISA_ERROR_IF(iter == _buffers.end(), "ray attribute not found: ", name);
        auto buffer = iter->second->view_as<T>();
        LUISA_ERROR_IF_NOT(buffer.size() == _capacity, "incorrect ray attribute buffer type");
        return buffer;
    }
};

}

#endif
