//
// Created by Mike on 8/27/2020.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <jitify/jitify.hpp>
#include <compute/device.h>

#include "check.h"
#include "cuda_codegen.h"
#include "cuda_buffer.h"
#include "cuda_jit_headers.h"

namespace luisa::cuda {

using luisa::compute::Buffer;
using luisa::compute::Device;
using luisa::compute::Dispatcher;
using luisa::compute::Kernel;
using luisa::compute::Texture;

class CudaDevice : public Device {

private:
    CUdevice _handle{};
    CUcontext _ctx{};
    uint _compute_capability{};

protected:
    std::shared_ptr<Buffer> _allocate_buffer(size_t size, size_t max_host_caches) override {
        return std::make_shared<CudaBuffer>(0, size, max_host_caches);
    }

    std::unique_ptr<Texture> _allocate_texture(uint32_t width, uint32_t height, compute::PixelFormat format, size_t max_caches) override {
        return std::unique_ptr<Texture>();
    }

    std::unique_ptr<Kernel> _compile_kernel(const compute::dsl::Function &function) override {
        std::ostringstream os;
        CudaCodegen codegen{os};
        codegen.emit(function);
        auto src = os.str();
        LUISA_INFO("Generated source:\n", src);
        
        auto &&headers = jitify::detail::get_jitsafe_headers_map();
        std::vector<const char *> header_names;
        std::vector<const char *> header_sources;
        header_names.reserve(headers.size());
        header_sources.reserve(headers.size());
        for (auto &&header_item : headers) {
            header_names.emplace_back(header_item.first.c_str());
            header_sources.emplace_back(header_item.second.c_str());
        }
        
        auto &&luisa_headers = get_jit_headers(_context);
        for (auto &&header_item : luisa_headers) {
            header_names.emplace_back(header_item.first);
            header_sources.emplace_back(header_item.second.c_str());
        }

        nvrtcProgram prog;
        nvrtcCreateProgram(&prog, src.c_str(), serialize(function.name(), ".cu").c_str(), header_sources.size(), header_sources.data(), header_names.data());// includeNames
        
        auto arch_opt = serialize("--gpu-architecture=compute_", _compute_capability);
        auto cuda_version_opt = serialize("-DCUDA_VERSION=", CUDART_VERSION);
        const char *opts[] = {
            arch_opt.c_str(),
            "--std=c++17",
            "--use_fast_math",
            "-default-device",
            "-restrict",
            "-ewp",
            "-dw",
            "-w",
            cuda_version_opt.c_str()};
        nvrtcCompileProgram(prog, sizeof(opts) / sizeof(const char *), opts);// options

        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char *log = new char[logSize];
        nvrtcGetProgramLog(prog, log);

        LUISA_INFO("Compile log: ", log);

        size_t ptxSize;
        nvrtcGetPTXSize(prog, &ptxSize);
        std::string ptx;
        ptx.resize(ptxSize);
        nvrtcGetPTX(prog, ptx.data());
        
        jitify::detail::ptx_remove_unused_globals(&ptx);
        LUISA_INFO("Generated PTX:\n", ptx);
        
        return std::unique_ptr<Kernel>();
    }

    void _launch(const std::function<void(Dispatcher &)> &dispatch) override {
    }

public:
    explicit CudaDevice(Context *context, uint32_t device_id);
    ~CudaDevice() noexcept override;

    void synchronize() override;
};

void CudaDevice::synchronize() {
    cuCtxSynchronize();
}

CudaDevice::CudaDevice(Context *context, uint32_t device_id) : Device{context} {

    static std::once_flag flag;
    std::call_once(flag, cuInit, 0);

    int count = 0;
    cuDeviceGetCount(&count);
    LUISA_ERROR_IF_NOT(device_id < count, "Invalid CUDA device index ", device_id, ": max available index is ", count - 1, ".");
    
    cuDeviceGet(&_handle, device_id);
    cuDevicePrimaryCtxRetain(&_ctx, _handle);
    
    char buffer[1024];
    cuDeviceGetName(buffer, 1024, _handle);
    auto major = 0;
    auto minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _handle);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _handle);
    _compute_capability = static_cast<uint>(major * 10 + minor);
    LUISA_INFO("Created CUDA device #", device_id, ", description: name = ", buffer, ", arch = sm_", _compute_capability, ".");
    
    nvrtcVersion(&major, &minor);
    LUISA_INFO("NVRTC version: ", major, ".", minor);
}

CudaDevice::~CudaDevice() noexcept {
    cuCtxSynchronize();
    cuDevicePrimaryCtxRelease(_handle);
}

}// namespace luisa::cuda

LUISA_EXPORT_DEVICE_CREATOR(luisa::cuda::CudaDevice)
