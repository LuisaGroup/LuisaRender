//
// Created by Mike Smith on 2020/2/19.
//

#pragma once

#include "data_types.h"

namespace luisa::interaction_attribute_flags {

LUISA_CONSTANT_SPACE constexpr auto POSITION_BIT = 0x01u;
LUISA_CONSTANT_SPACE constexpr auto NORMAL_BIT = 0x02u;
LUISA_CONSTANT_SPACE constexpr auto UV_BIT = 0x04u;
LUISA_CONSTANT_SPACE constexpr auto MATERIAL_INFO_BIT = 0x08u;
LUISA_CONSTANT_SPACE constexpr auto WO_AND_DISTANCE_BIT = 0x10u;

LUISA_CONSTANT_SPACE constexpr auto ALL_BITS = POSITION_BIT | NORMAL_BIT | UV_BIT | MATERIAL_INFO_BIT | WO_AND_DISTANCE_BIT;

}

#ifndef LUISA_DEVICE_COMAPTIBLE

#include "material.h"

namespace luisa {

class InteractionBufferSet {

private:
    size_t _size{0ul};
    uint _attribute_flags{0x0u};
    
    std::unique_ptr<Buffer<float3>> _position_buffer;
    std::unique_ptr<Buffer<float3>> _normal_buffer;
    std::unique_ptr<Buffer<float2>> _uv_buffer;
    std::unique_ptr<Buffer<MaterialInfo>> _material_info_buffer;
    std::unique_ptr<Buffer<float4>> _wo_and_distance_buffer;

public:
    InteractionBufferSet() noexcept = default;
    
    InteractionBufferSet(Device *device, size_t capacity, uint flags = interaction_attribute_flags::ALL_BITS)
        : _size{capacity},
          _attribute_flags{flags},
          _position_buffer{(flags & interaction_attribute_flags::POSITION_BIT) ? device->create_buffer<float3>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _normal_buffer{(flags & interaction_attribute_flags::NORMAL_BIT) ? device->create_buffer<float3>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _uv_buffer{(flags & interaction_attribute_flags::UV_BIT) ? device->create_buffer<float2>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _material_info_buffer{(flags & interaction_attribute_flags::MATERIAL_INFO_BIT) ?
                                device->create_buffer<MaterialInfo>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _wo_and_distance_buffer{(flags & interaction_attribute_flags::WO_AND_DISTANCE_BIT) ?
                                  device->create_buffer<float4>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr} {}
    
    [[nodiscard]] size_t size() const noexcept { return _size; }
    [[nodiscard]] uint attribute_flags() const noexcept { return _attribute_flags; }
    
    [[nodiscard]] bool has_position_buffer() const noexcept { return (_attribute_flags & interaction_attribute_flags::POSITION_BIT) != 0u; }
    [[nodiscard]] bool has_normal_buffer() const noexcept { return (_attribute_flags & interaction_attribute_flags::NORMAL_BIT) != 0u; }
    [[nodiscard]] bool has_uv_buffer() const noexcept { return (_attribute_flags & interaction_attribute_flags::UV_BIT) != 0u; }
    [[nodiscard]] bool has_material_info_buffer() const noexcept { return (_attribute_flags & interaction_attribute_flags::MATERIAL_INFO_BIT) != 0u; }
    [[nodiscard]] bool has_wo_and_distance_buffer() const noexcept { return (_attribute_flags & interaction_attribute_flags::WO_AND_DISTANCE_BIT) != 0u; }
    
    [[nodiscard]] auto position_buffer() const noexcept {
        LUISA_ERROR_IF_NOT(has_position_buffer(), "no position buffer present");
        return _position_buffer->view();
    }
    
    [[nodiscard]] auto normal_buffer() const noexcept {
        LUISA_ERROR_IF_NOT(has_normal_buffer(), "no normal buffer present");
        return _normal_buffer->view();
    }
    
    [[nodiscard]] auto uv_buffer() const noexcept {
        LUISA_ERROR_IF_NOT(has_uv_buffer(), "no uv buffer present");
        return _uv_buffer->view();
    }
    
    [[nodiscard]] auto material_info_buffer() const noexcept {
        LUISA_ERROR_IF_NOT(has_material_info_buffer(), "no material info buffer present");
        return _material_info_buffer->view();
    }
    
    [[nodiscard]] auto wo_and_distance_buffer() const noexcept {
        LUISA_ERROR_IF_NOT(has_wo_and_distance_buffer(), "no wo and distance buffer present");
        return _wo_and_distance_buffer->view();
    }
};

}

#endif
