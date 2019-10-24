//
// Created by Mike Smith on 2019/10/24.
//

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <util/resource_manager.h>
#include <util/string_manipulation.h>
#include "metal_device.h"

struct MetalDeviceWrapper { id<MTLDevice> device; };
struct MetalLibraryWrapper { id<MTLLibrary> library; };

MetalDevice::MetalDevice()
    : _device_wrapper{std::make_unique<MetalDeviceWrapper>()},
      _library_wrapper{std::make_unique<MetalLibraryWrapper>()} {
    
    _device_wrapper->device = MTLCreateSystemDefaultDevice();
    
    auto library_path = make_objc_string(ResourceManager::instance().binary_path("kernels.metallib").c_str());
    _library_wrapper->library = [_device_wrapper->device newLibraryWithFile:library_path error:nullptr];
    
}

std::shared_ptr<Kernel> MetalDevice::create_kernel(std::string_view function_name) {

}
