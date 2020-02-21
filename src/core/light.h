//
// Created by Mike Smith on 2020/2/5.
//

#pragma once

#include "ray.h"
#include "node.h"
#include "parser.h"
#include "device.h"

namespace luisa {

class LightSampleBufferSet {

private:
    std::unique_ptr<Buffer<float4>> _radiance_and_pdf_w_buffer;
    std::unique_ptr<Buffer<bool>> _is_delta_buffer;
    std::unique_ptr<Buffer<Ray>> _shadow_ray_buffer;

public:
    LightSampleBufferSet(Device *device, size_t capacity)
        : _radiance_and_pdf_w_buffer{device->create_buffer<float4>(capacity, BufferStorage::DEVICE_PRIVATE)},
          _is_delta_buffer{device->create_buffer<bool>(capacity, BufferStorage::DEVICE_PRIVATE)},
          _shadow_ray_buffer{device->create_buffer<Ray>(capacity, BufferStorage::DEVICE_PRIVATE)} {}
    
};

class Light : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Light);

protected:
    inline static auto _used_tag_count = 0u;

public:
    Light(Device *device, const ParameterSet &parameter_set[[maybe_unused]]) : Node{device} {}
    [[nodiscard]] static uint used_tag_count() noexcept { return _used_tag_count; }
    [[nodiscard]] virtual uint tag() const noexcept = 0;
    [[nodiscard]] virtual std::unique_ptr<Kernel> create_generate_samples_kernel() = 0;
    [[nodiscard]] virtual size_t data_stride() const noexcept = 0;
    [[nodiscard]] virtual size_t sample_dimensions() const noexcept = 0;
    virtual void encode_data(TypelessBuffer &buffer, size_t index) = 0;
};
    
}
