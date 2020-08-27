//
// Created by Mike on 8/27/2020.
//

#include <compute/device.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include "check.h"
#include "cuda_codegen.h"

namespace luisa::cuda {

using luisa::compute::Buffer;
using luisa::compute::Device;
using luisa::compute::Dispatcher;
using luisa::compute::Kernel;
using luisa::compute::Texture;

class CudaDevice : public Device {

private:
    CUdevice _handle;

protected:
    std::unique_ptr<Buffer> _allocate_buffer(size_t size, size_t max_host_caches) override {
        return std::unique_ptr<Buffer>();
    }

    std::unique_ptr<Texture> _allocate_texture(uint32_t width, uint32_t height, compute::PixelFormat format, size_t max_caches) override {
        return std::unique_ptr<Texture>();
    }

    std::unique_ptr<Kernel> _compile_kernel(const compute::dsl::Function &function) override {
        LUISA_INFO("Hello!");
        std::ostringstream os;
        CudaCodegen codegen{os};
        codegen.emit(function);
        LUISA_INFO("Generated source:\n", os.str());
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

    const char *saxpy = "extern \"C\" __global__                                         \n"
                        "void saxpy(float a, float *x, float *y, float *out, size_t n)   \n"
                        "{                                                               \n"
                        "  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;           \n"
                        "  if (auto t = tid; t < n) {                                                \n"
                        "    out[tid] = a * x[tid] + y[tid];                             \n"
                        "  }                                                             \n"
                        "}                                                               \n";

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog,   // prog
                       saxpy,   // buffer
                       "foo.cu",// name
                       0,       // numHeaders
                       NULL,    // headers
                       NULL);   // includeNames

    const char *opts[] = {"--gpu-architecture=compute_80",
                          "--fmad=false",
                          "--std=c++17"};
    nvrtcCompileProgram(prog, // prog
                        3,    // numOptions
                        opts);// options

    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    char *log = new char[logSize];
    nvrtcGetProgramLog(prog, log);

    LUISA_INFO("Compile log: ", log);

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    static std::once_flag flag;
    std::call_once(flag, cuInit, 0);
    
    int count = 0;
    cuDeviceGetCount(&count);
    for (auto i = 0; i < count; i++) {
        cuDeviceGet(&_handle, i);
        char buffer[1024];
        cuDeviceGetName(buffer, 1024, _handle);
        LUISA_INFO("CUDA Device #", i, ": ", buffer);
    }

    int major = 0;
    int minor = 0;
    nvrtcVersion(&major, &minor);
    LUISA_INFO("NVRTC Version: ", major, ".", minor);
    auto path = std::filesystem::canonical(getenv("OptiX_INSTALL_DIR"));
    LUISA_INFO("OptiX include directory: ", path);
}

}// namespace luisa::cuda

LUISA_EXPORT_DEVICE_CREATOR(luisa::cuda::CudaDevice)
