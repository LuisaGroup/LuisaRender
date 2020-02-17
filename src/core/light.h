//
// Created by Mike Smith on 2020/2/5.
//

#pragma once

#include "data_types.h"
#include "node.h"
#include "parser.h"
#include "device.h"

namespace luisa {

struct LightSampleBufferSetView {
    BufferView<float4> Li_and_pdf_w_buffer;
    BufferView<bool> is_delta_buffer;
};

struct LightSampleBufferSet {
    
    std::unique_ptr<Buffer<float4>> Li_and_pdf_w_buffer;
    std::unique_ptr<Buffer<bool>> is_delta_buffer;
    
    LightSampleBufferSet(Device *device, size_t capacity)
        : Li_and_pdf_w_buffer{device->create_buffer<float4>(capacity, BufferStorage::DEVICE_PRIVATE)},
          is_delta_buffer{device->create_buffer<bool>(capacity, BufferStorage::DEVICE_PRIVATE)} {}
    
    [[nodiscard]] auto view() { return LightSampleBufferSetView{Li_and_pdf_w_buffer->view(), is_delta_buffer->view()}; }
};

class Light : public Node {

private:
    LUISA_MAKE_NODE_CREATOR_REGISTRY(Light);

protected:
    inline static auto _used_tag_count = 0u;
    [[nodiscard]] virtual uint _assign_tag() const noexcept = 0;

protected:
    uint _tag = _assign_tag();

public:
    Light(Device *device, const ParameterSet &parameter_set[[maybe_unused]]) : Node{device} {}
    [[nodiscard]] static uint used_tag_count() noexcept { return _used_tag_count; }
    [[nodiscard]] uint tag() const noexcept { return _tag; }
    [[nodiscard]] virtual std::unique_ptr<Kernel> create_sample_kernel() = 0;
    [[nodiscard]] virtual size_t data_stride() const noexcept = 0;
    virtual void encode_data(TypelessBuffer &buffer, size_t index) = 0;
};

#define LUISA_MAKE_LIGHT_TAG_ASSIGNMENT(name)                                  \
    [[nodiscard]] uint _assign_tag() const noexcept override {                 \
        static auto t = _used_tag_count++;                                     \
        std::cout << "[INFO] assigned tag " << t << " to " name << std::endl;  \
        return t;                                                              \
    }
    
}
