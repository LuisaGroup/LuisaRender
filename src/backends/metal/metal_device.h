//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include <core/device.h>

namespace luisa::metal {

struct MetalDeviceWrapper;
struct MetalLibraryWrapper;
struct MetalCommandQueueWrapper;
struct MetalFunctionWrapper;

class MetalDevice : public Device {

private:
    std::unique_ptr<MetalDeviceWrapper> _device_wrapper;
    std::unique_ptr<MetalLibraryWrapper> _library_wrapper;
    std::unique_ptr<MetalCommandQueueWrapper> _command_queue_wrapper;
    std::unordered_map<std::string, std::unique_ptr<MetalFunctionWrapper>> _function_wrappers;
    
    LUISA_DEVICE_CREATOR("Metal") { return std::make_unique<MetalDevice>(); }

protected:
    void _launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) override;

public:
    MetalDevice();
    ~MetalDevice() noexcept override = default;
    std::unique_ptr<Kernel> create_kernel(std::string_view function_name) override;
    std::unique_ptr<Acceleration> create_acceleration(Geometry &geometry) override;
    std::unique_ptr<TypelessBuffer> allocate_buffer(size_t capacity, BufferStorage storage) override;
    
    void launch(std::function<void(KernelDispatcher &)> dispatch) override;
};

}
