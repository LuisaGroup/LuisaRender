//
// Created by Mike Smith on 2020/1/31.
//

#pragma once

#include "data_types.h"

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
