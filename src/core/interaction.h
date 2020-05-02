//
// Created by Mike Smith on 2020/2/19.
//

#pragma once

#include "data_types.h"

namespace luisa::interaction {

namespace attribute {

LUISA_CONSTANT_SPACE constexpr auto POSITION = 0x01u;
LUISA_CONSTANT_SPACE constexpr auto NORMAL = 0x02u;
LUISA_CONSTANT_SPACE constexpr auto UV = 0x04u;
LUISA_CONSTANT_SPACE constexpr auto WO_AND_PDF = 0x08u;
LUISA_CONSTANT_SPACE constexpr auto INSTANCE_ID = 0x10u;
LUISA_CONSTANT_SPACE constexpr auto EMISSION = 0x20u;
LUISA_CONSTANT_SPACE constexpr auto SCATTERING = 0x40u;
LUISA_CONSTANT_SPACE constexpr auto WI_AND_PDF = 0x80u;

LUISA_CONSTANT_SPACE constexpr auto ALL = POSITION | NORMAL | UV | WO_AND_PDF | INSTANCE_ID | EMISSION | SCATTERING | WI_AND_PDF;

}

namespace state {

LUISA_CONSTANT_SPACE constexpr auto MISS = static_cast<uint8_t>(0x00u);

LUISA_CONSTANT_SPACE constexpr auto HIT = static_cast<uint8_t>(0x01u);
LUISA_CONSTANT_SPACE constexpr auto BACK_FACE = static_cast<uint8_t>(0x02u);
LUISA_CONSTANT_SPACE constexpr auto EMISSIVE = static_cast<uint8_t>(0x04u);
LUISA_CONSTANT_SPACE constexpr auto DELTA_LIGHT = static_cast<uint8_t>(0x08u);
LUISA_CONSTANT_SPACE constexpr auto SPECULAR_BSDF = static_cast<uint8_t>(0x10u);

}

}

#ifndef LUISA_DEVICE_COMPATIBLE

#include "material.h"

namespace luisa {

class InteractionBufferSet {

private:
    size_t _size{0ul};
    uint _attribute_flags{0x0u};
    
    std::unique_ptr<Buffer<uint8_t>> _state_buffer;
    std::unique_ptr<Buffer<float3>> _position_buffer;
    std::unique_ptr<Buffer<float3>> _normal_buffer;
    std::unique_ptr<Buffer<float2>> _uv_buffer;
    std::unique_ptr<Buffer<float4>> _wo_and_pdf_buffer;
    std::unique_ptr<Buffer<uint>> _instance_id_buffer;
    std::unique_ptr<Buffer<float3>> _emission_buffer;
    std::unique_ptr<Buffer<float3>> _scattering_buffer;
    std::unique_ptr<Buffer<float4>> _wi_and_pdf_buffer;

public:
    InteractionBufferSet() noexcept = default;
    
    InteractionBufferSet(Device *device, size_t capacity, uint flags = interaction::attribute::ALL)
        : _size{capacity},
          _attribute_flags{flags},
          _state_buffer{device->create_buffer<uint8_t>(capacity, BufferStorage::DEVICE_PRIVATE)},
          _position_buffer{(flags & interaction::attribute::POSITION) ? device->create_buffer<float3>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _normal_buffer{(flags & interaction::attribute::NORMAL) ? device->create_buffer<float3>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _uv_buffer{(flags & interaction::attribute::UV) ? device->create_buffer<float2>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _wo_and_pdf_buffer{(flags & interaction::attribute::WO_AND_PDF) ? device->create_buffer<float4>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _instance_id_buffer{(flags & interaction::attribute::INSTANCE_ID) ? device->create_buffer<uint>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _emission_buffer{(flags & interaction::attribute::EMISSION) ? device->create_buffer<float3>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _scattering_buffer{(flags & interaction::attribute::SCATTERING) ? device->create_buffer<float3>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr},
          _wi_and_pdf_buffer{(flags & interaction::attribute::WI_AND_PDF) ? device->create_buffer<float4>(capacity, BufferStorage::DEVICE_PRIVATE) : nullptr} {}
    
    [[nodiscard]] size_t size() const noexcept { return _size; }
    [[nodiscard]] uint attribute_flags() const noexcept { return _attribute_flags; }
    
    [[nodiscard]] bool has_position_buffer() const noexcept { return (_attribute_flags & interaction::attribute::POSITION) != 0u; }
    [[nodiscard]] bool has_normal_buffer() const noexcept { return (_attribute_flags & interaction::attribute::NORMAL) != 0u; }
    [[nodiscard]] bool has_uv_buffer() const noexcept { return (_attribute_flags & interaction::attribute::UV) != 0u; }
    [[nodiscard]] bool has_wo_and_pdf_buffer() const noexcept { return (_attribute_flags & interaction::attribute::WO_AND_PDF) != 0u; }
    [[nodiscard]] bool has_wi_and_pdf_buffer() const noexcept { return (_attribute_flags & interaction::attribute::WI_AND_PDF) != 0u; }
    [[nodiscard]] bool has_instance_id_buffer() const noexcept { return (_attribute_flags & interaction::attribute::INSTANCE_ID) != 0u; }
    [[nodiscard]] bool has_emission_buffer() const noexcept { return (_attribute_flags & interaction::attribute::EMISSION) != 0u; }
    [[nodiscard]] bool has_scattering_buffer() const noexcept { return (_attribute_flags & interaction::attribute::SCATTERING) != 0u; }
    
    [[nodiscard]] auto position_buffer() const noexcept {
        LUISA_EXCEPTION_IF_NOT(has_position_buffer(), "No position buffer present");
        return _position_buffer->view();
    }
    
    [[nodiscard]] auto normal_buffer() const noexcept {
        LUISA_EXCEPTION_IF_NOT(has_normal_buffer(), "No normal buffer present");
        return _normal_buffer->view();
    }
    
    [[nodiscard]] auto uv_buffer() const noexcept {
        LUISA_EXCEPTION_IF_NOT(has_uv_buffer(), "No UV buffer present");
        return _uv_buffer->view();
    }
    
    [[nodiscard]] auto wo_and_pdf_buffer() const noexcept {
        LUISA_EXCEPTION_IF_NOT(has_wo_and_pdf_buffer(), "No wo and pdf buffer present");
        return _wo_and_pdf_buffer->view();
    }
    
    [[nodiscard]] auto wi_and_pdf_buffer() const noexcept {
        LUISA_EXCEPTION_IF_NOT(has_wi_and_pdf_buffer(), "No wi and pdf buffer present");
        return _wi_and_pdf_buffer->view();
    }
    
    [[nodiscard]] auto instance_id_buffer() const noexcept {
        LUISA_EXCEPTION_IF_NOT(has_instance_id_buffer(), "No instance id buffer present");
        return _instance_id_buffer->view();
    }
    
    [[nodiscard]] auto emission_buffer() const noexcept {
        LUISA_EXCEPTION_IF_NOT(has_emission_buffer(), "No instance emission present");
        return _emission_buffer->view();
    }
    
    [[nodiscard]] auto scattering_buffer() const noexcept {
        LUISA_EXCEPTION_IF_NOT(has_scattering_buffer(), "No instance scattering present");
        return _scattering_buffer->view();
    }
    
    [[nodiscard]] auto state_buffer() const noexcept {
        return _state_buffer->view();
    }
};

}

#endif
