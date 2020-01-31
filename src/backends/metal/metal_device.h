//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <memory>
#include <core/device.h>

namespace luisa::metal {

class MetalDevice : public Device {

private:
    std::unique_ptr<struct MetalDeviceWrapper> _device_wrapper;
    std::unique_ptr<struct MetalLibraryWrapper> _library_wrapper;
    std::unique_ptr<struct MetalCommandQueueWrapper> _command_queue_wrapper;
    
    DEVICE_CREATOR("Metal") { return std::make_unique<MetalDevice>(); }

public:
    MetalDevice();
    std::unique_ptr<Kernel> create_kernel(std::string_view function_name) override;
    std::unique_ptr<Texture> create_texture(uint2 size, TextureFormatTag format_tag, TextureAccessTag access_tag) override;
    std::unique_ptr<Acceleration> create_acceleration(Buffer &position_buffer, size_t stride, size_t triangle_count) override;
    std::unique_ptr<Buffer> create_buffer(size_t capacity, BufferStorage storage) override;
    
    void launch(std::function<void(KernelDispatcher &)> dispatch) override;
    void launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) override;
    
};

}
