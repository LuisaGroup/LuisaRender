//
// Created by Mike Smith on 2020/2/5.
//

#pragma once

#include "ray.h"
#include "selection.h"
#include "node.h"
#include "parser.h"
#include "device.h"
#include "interaction.h"

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
    
    [[nodiscard]] auto radiance_and_pdf_w_buffer() { return _radiance_and_pdf_w_buffer->view(); }
    [[nodiscard]] auto is_delta_buffer() { return _is_delta_buffer->view(); }
    [[nodiscard]] auto shadow_ray_buffer() { return _shadow_ray_buffer->view(); }
};

class Light : public Node {

public:
    static constexpr auto MAX_LIGHT_TAG_COUNT = 16u;
    
    using SampleLightsDispatch = std::function<void(
        KernelDispatcher &dispatch, Kernel &kernel, uint dispatch_extent, BufferView<float> sample_buffer,
        TypelessBuffer &light_data_buffer, BufferView<Selection> queue, BufferView<uint> queue_size,
        InteractionBufferSet &interactions, Geometry *geometry, LightSampleBufferSet &light_samples)>;

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Light);

protected:
    static uint _assign_tag() {
        static auto next_tag = 0u;
        LUISA_ERROR_IF(next_tag == MAX_LIGHT_TAG_COUNT, "too many light tags assigned, limit: ", MAX_LIGHT_TAG_COUNT);
        return next_tag++;
    }

public:
    Light(Device *device, const ParameterSet &parameter_set[[maybe_unused]]) : Node{device} {}
    [[nodiscard]] virtual uint tag() const noexcept = 0;
    [[nodiscard]] virtual std::unique_ptr<Kernel> create_generate_samples_kernel() = 0;
    [[nodiscard]] virtual SampleLightsDispatch create_generate_samples_dispatch() = 0;
    [[nodiscard]] virtual size_t data_stride() const noexcept = 0;
    [[nodiscard]] virtual Shape *shape() const noexcept { return nullptr; }
    [[nodiscard]] virtual uint sampling_dimensions() const noexcept = 0;
    virtual void encode_data(TypelessBuffer &buffer, size_t index) = 0;
};
    
}
