//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <compute/buffer.h>
#include <compute/device.h>
#include <render/material.h>

namespace luisa::render {

using compute::Device;
using compute::BufferView;
using compute::Expr;

class InteractionBuffers {

public:
    enum Component {
        COMPONENT_MISS = 1u << 0u,
        COMPONENT_PI = 1u << 1u,
        COMPONENT_WO = 1u << 2u,
        COMPONENT_NG = 1u << 3u,
        COMPONENT_NS = 1u << 4u,
        COMPONENT_UV = 1u << 5u,
        COMPONENT_DISTANCE = 1u << 6u,
        COMPONENT_MATERIAL = 1u << 7u,
        COMPONENT_PDF = 1u << 8u,
        COMPONENT_SHADER = 1u << 9u,
        
        COMPONENT_NONE = 0u,
        COMPONENT_ALL = 0xffffffffu
    };
    
    struct ShaderBuffers {
        BufferView<uint> type;
        BufferView<uint> index;
        BufferView<float> prob;
        BufferView<float> weight;
    };
    
private:
    size_t _size{0u};
    uint32_t _components{COMPONENT_NONE};
    
    BufferView<bool> _miss;
    BufferView<float3> _pi;
    
    BufferView<float3> _wo;
    BufferView<float> _distance;
    
    BufferView<float3> _ng;
    BufferView<float3> _ns;
    BufferView<float2> _uv;
    BufferView<MaterialHandle> _material;
    
    BufferView<float> _pdf;
    
    ShaderBuffers _shader;

public:
    void create(Device *device, size_t size, uint32_t components = COMPONENT_ALL) noexcept {
        _size = size;
        _components = static_cast<Component>(components);
        _miss = has_miss() ? device->allocate_buffer<bool>(size) : BufferView<bool>{};
        _pi = has_pi() ? device->allocate_buffer<float3>(size) : BufferView<float3>{};
        _wo = has_wo() ? device->allocate_buffer<float3>(size) : BufferView<float3>{};
        _distance = has_distance() ? device->allocate_buffer<float>(size) : BufferView<float>{};
        _ng = has_ng() ? device->allocate_buffer<float3>(size) : BufferView<float3>{};
        _ns = has_ns() ? device->allocate_buffer<float3>(size) : BufferView<float3>{};
        _uv = has_uv() ? device->allocate_buffer<float2>(size) : BufferView<float2>{};
        _material = has_material() ? device->allocate_buffer<MaterialHandle>(size) : BufferView<MaterialHandle>{};
        _pdf = has_pdf() ? device->allocate_buffer<float>(size) : BufferView<float>{};
        
        if (has_shader()) {
            _shader.type = device->allocate_buffer<uint>(size);
            _shader.index = device->allocate_buffer<uint>(size);
            _shader.prob = device->allocate_buffer<float>(size);
            _shader.weight = device->allocate_buffer<float>(size);
        }
    }
    
    [[nodiscard]] bool has_miss() const noexcept { return _components & COMPONENT_MISS; }
    [[nodiscard]] bool has_pi() const noexcept { return _components & COMPONENT_PI; }
    [[nodiscard]] bool has_wo() const noexcept { return _components & COMPONENT_WO; }
    [[nodiscard]] bool has_distance() const noexcept { return _components & COMPONENT_DISTANCE; }
    [[nodiscard]] bool has_ng() const noexcept { return _components & COMPONENT_NG; }
    [[nodiscard]] bool has_ns() const noexcept { return _components & COMPONENT_NS; }
    [[nodiscard]] bool has_uv() const noexcept { return _components & COMPONENT_UV; }
    [[nodiscard]] bool has_material() const noexcept { return _components & COMPONENT_MATERIAL; }
    [[nodiscard]] bool has_pdf() const noexcept { return _components & COMPONENT_PDF; }
    [[nodiscard]] bool has_shader() const noexcept { return _components & COMPONENT_SHADER; }
    
    [[nodiscard]] auto &miss() noexcept { return _miss; }
    [[nodiscard]] auto &pi() noexcept { return _pi; }
    [[nodiscard]] auto &wo() noexcept { return _wo; }
    [[nodiscard]] auto &distance() noexcept { return _distance; }
    [[nodiscard]] auto &ng() noexcept { return _ng; }
    [[nodiscard]] auto &ns() noexcept { return _ns; }
    [[nodiscard]] auto &uv() noexcept { return _uv; }
    [[nodiscard]] auto &material() noexcept { return _material; }
    [[nodiscard]] auto &pdf() noexcept { return _pdf; }
    [[nodiscard]] auto &shader() noexcept { return _shader; }
    
    [[nodiscard]] const auto &miss() const noexcept { return _miss; }
    [[nodiscard]] const auto &pi() const noexcept { return _pi; }
    [[nodiscard]] const auto &wo() const noexcept { return _wo; }
    [[nodiscard]] const auto &distance() const noexcept { return _distance; }
    [[nodiscard]] const auto &ng() const noexcept { return _ng; }
    [[nodiscard]] const auto &ns() const noexcept { return _ns; }
    [[nodiscard]] const auto &uv() const noexcept { return _uv; }
    [[nodiscard]] const auto &material() const noexcept { return _material; }
    [[nodiscard]] const auto &pdf() const noexcept { return _pdf; }
    [[nodiscard]] const auto &shader() const noexcept { return _shader; }
    
    [[nodiscard]] bool empty() const noexcept { return _size == 0u || _components == COMPONENT_NONE; }
    [[nodiscard]] size_t size() const noexcept { return _size; }
};

}
