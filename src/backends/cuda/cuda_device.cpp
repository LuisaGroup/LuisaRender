//
// Created by Mike on 8/27/2020.
//

#include <cuda.h>
#include <nvrtc.h>
#include <compute/device.h>

namespace luisa::cuda {

using luisa::compute::Device;
using luisa::compute::Kernel;
using luisa::compute::Buffer;
using luisa::compute::Texture;
using luisa::compute::Dispatcher;

class CudaDevice : public Device {

private:


protected:
    std::unique_ptr<Buffer> _allocate_buffer(size_t size, size_t max_host_caches) override {
        return std::unique_ptr<Buffer>();
    }
    
    std::unique_ptr<Texture> _allocate_texture(uint32_t width, uint32_t height, compute::PixelFormat format, size_t max_caches) override {
        return std::unique_ptr<Texture>();
    }
    
    std::unique_ptr<Kernel> _compile_kernel(const compute::dsl::Function &function) override {
        return std::unique_ptr<Kernel>();
    }
    
    void _launch(const std::function<void(Dispatcher &)> &dispatch) override {
    }

public:
    explicit CudaDevice(Context *context);
    ~CudaDevice() noexcept override = default;
    
    void synchronize() override;
};

void CudaDevice::synchronize() {

}

CudaDevice::CudaDevice(Context *context) : Device{context} {
    int major = 0;
    int minor = 0;
    nvrtcVersion(&major, &minor);
    LUISA_INFO("NVRTC Version: ", major, ".", minor);
}

}

LUISA_EXPORT_DEVICE_CREATOR(luisa::cuda::CudaDevice)
