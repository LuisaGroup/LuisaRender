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

enum struct RayState : uint8_t {
    UNINITIALIZED,
    GENERATED,
    TRACED,
    EXTENDED,
    FINISHED,
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
    
    template<typename T>
    BufferView<T> attribute_buffer(const std::string &name) {
        return _attribute_buffers.at(name)->view<T>();
    }
};

struct RayQueueView {
    
    BufferView<uint> index_buffer;
    BufferView<uint> size_buffer;
    
    [[nodiscard]] size_t capacity() const noexcept { return index_buffer.element_count(); }
};

class RayQueueManager : public Noncopyable {

private:
    Device *_device;
    std::unique_ptr<Buffer> _queue_size_buffer;
    std::vector<std::unique_ptr<Buffer>> _queue_index_buffers;

public:
    RayQueueManager(Device *device, size_t max_queue_count)
        : _device{device},
          _queue_size_buffer{device->create_buffer<uint>(max_queue_count, BufferStorage::DEVICE_PRIVATE)} {}
    
    [[nodiscard]] RayQueueView allocate_queue(size_t capacity) {
        assert(_queue_index_buffers.size() < _queue_size_buffer->view<uint>().element_count());
        auto size_buffer_view = _queue_size_buffer->view<uint>(_queue_index_buffers.size());
        auto index_buffer_view = _queue_index_buffers.emplace_back(_device->create_buffer<uint>(capacity, BufferStorage::DEVICE_PRIVATE))->view<uint>();
        return {index_buffer_view, size_buffer_view};
    }
    
    [[nodiscard]] size_t allocated_queue_count() const noexcept { return _queue_index_buffers.size(); }
    [[nodiscard]] BufferView<uint> allocated_queue_size_buffer() { return _queue_size_buffer->view<uint>(0ul, _queue_index_buffers.size()); }
};

}

#endif
