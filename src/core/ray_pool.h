//
// Created by Mike Smith on 2020/1/31.
//

#pragma once

#include <string>
#include <unordered_map>

#include "ray.h"
#include "device.h"
#include "buffer.h"

namespace luisa {

class RayPool {

private:
    Device *_device;
    size_t _capacity;
    std::unordered_map<std::string, std::unique_ptr<Buffer>> _attribute_buffers;

public:
    RayPool(Device *device, size_t capacity) : _device{device}, _capacity{capacity} {}
    
    [[nodiscard]] static std::unique_ptr<RayPool> create(Device *device, size_t capacity) noexcept {
        return std::make_unique<RayPool>(device, capacity);
    }
    
    [[nodiscard]] size_t capacity() const noexcept { return _capacity; }
    
    template<typename T>
    void add_attribute_buffer(const std::string &name) {
        assert(_attribute_buffers.count(name) == 0ul);
        _attribute_buffers.emplace(name, _device->create_buffer(_capacity * sizeof(T), BufferStorage::DEVICE_PRIVATE));
    }
    
    Buffer &attribute_buffer(const std::string &name) {
        return *_attribute_buffers.at(name);
    }
    
};

}
