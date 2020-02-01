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

enum struct RayState : uint8_t {
    UNINITIALIZED,
    GENERATED,
    TRACED,
    SHADED,
    INVALIDATED
};

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include <string>
#include <vector>
#include <unordered_map>

#include <util/noncopyable.h>

#include "device.h"

namespace luisa {

class RayPool : public Noncopyable {

private:
    Device *_device;
    size_t _capacity;
    std::unordered_map<std::string, std::unique_ptr<Buffer>> _attribute_buffers;

public:
    RayPool(Device *device, size_t capacity) : _device{device}, _capacity{capacity} {}
    
    [[nodiscard]] static std::unique_ptr<RayPool> create(Device *device, size_t capacity) {
        return std::make_unique<RayPool>(device, capacity);
    }
    
    [[nodiscard]] size_t capacity() const noexcept { return _capacity; }
    
    template<typename T>
    void add_attribute_buffer(const std::string &name) {
        assert(_attribute_buffers.count(name) == 0ul);
        _attribute_buffers.emplace(name, _device->create_buffer<T>(_capacity, BufferStorage::DEVICE_PRIVATE));
    }
    
    Buffer &attribute_buffer(const std::string &name) {
        return *_attribute_buffers.at(name);
    }
    
    Buffer &operator[](const std::string &attribute_name) {
        return attribute_buffer(attribute_name);
    }
};

}

#endif
