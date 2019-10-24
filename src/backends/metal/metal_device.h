//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <memory>
#include <core/device.h>

struct MetalDeviceWrapper;
struct MetalLibraryWrapper;

class MetalDevice : public Device {

private:
    std::unique_ptr<MetalDeviceWrapper> _device_wrapper;
    std::unique_ptr<MetalLibraryWrapper> _library_wrapper;
    
    DEVICE_CREATOR("Metal") { return std::make_shared<MetalDevice>(); }

public:
    MetalDevice();

public:
    std::shared_ptr<Kernel> create_kernel(std::string_view function_name) override;
    
    std::shared_ptr<Texture> create_texture() override {
        return std::shared_ptr<Texture>();
    }
    
};
