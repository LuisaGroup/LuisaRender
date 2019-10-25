//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <memory>
#include <core/device.h>

class MetalDevice : public Device {

private:
    std::unique_ptr<struct MetalDeviceWrapper> _device_wrapper;
    std::unique_ptr<struct MetalLibraryWrapper> _library_wrapper;
    std::unique_ptr<struct MetalCommandQueueWrapper> _command_queue_wrapper;
    
    DEVICE_CREATOR("Metal") { return std::make_shared<MetalDevice>(); }

public:
    MetalDevice();
    std::shared_ptr<Kernel> create_kernel(std::string_view function_name) override;
    std::shared_ptr<Texture> create_texture() override;
    
    void launch(std::function<void(KernelDispatcher &)> dispatch) override;
    void launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) override;
    
};
