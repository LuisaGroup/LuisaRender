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
    
    LUISA_DEVICE_CREATOR("Metal") { return std::make_unique<MetalDevice>(); }

public:
    MetalDevice();
    std::unique_ptr<Kernel> create_kernel(std::string_view function_name) override;
    std::unique_ptr<Acceleration> create_acceleration(Geometry &geometry) override;
    std::unique_ptr<TypelessBuffer> allocate_buffer(size_t capacity, BufferStorage storage) override;
    
    void launch(std::function<void(KernelDispatcher &)> dispatch) override;
    void launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) override;
    
};

}
