//
// Created by Mike on 8/27/2020.
//

#include <cuda.h>
#include <nvrtc.h>

#include <jitify/jitify.hpp>

#include <compute/device.h>
#include <core/sha1.h>

#include "cuda_buffer.h"
#include "cuda_check.h"
#include "cuda_codegen.h"
#include "cuda_dispatcher.h"
#include "cuda_jit_headers.h"
#include "cuda_kernel.h"

namespace luisa::cuda {

using luisa::compute::Buffer;
using luisa::compute::Device;
using luisa::compute::Dispatcher;
using luisa::compute::Kernel;
using luisa::compute::Texture;

class CudaDevice : public Device {

public:
    static constexpr auto max_command_queue_size = 16u;

private:
    CUdevice _handle{};
    CUcontext _ctx{};
    CUstream _stream{};
    CUevent _sync_event{};
    std::vector<CUmodule> _modules;
    std::map<SHA1::Digest, CUfunction> _kernel_cache;
    
    std::mutex _dispatch_mutex;
    std::condition_variable _dispatch_cv;
    std::queue<std::unique_ptr<CudaDispatcher>> _dispatch_queue;
    std::thread _dispatch_thread;
    std::atomic<bool> _stop_signal{false};

    uint32_t _compute_capability{};

protected:
    std::shared_ptr<Buffer> _allocate_buffer(size_t size) override;

    std::unique_ptr<Texture> _allocate_texture(uint32_t width, uint32_t height, compute::PixelFormat format) override {
        LUISA_EXCEPTION("Not implemented!");
    }

    std::unique_ptr<Kernel> _compile_kernel(const compute::dsl::Function &function) override;

    void _launch(const std::function<void(Dispatcher &)> &dispatch) override;

public:
    explicit CudaDevice(Context *context, uint32_t device_id);
    ~CudaDevice() noexcept override;
    void synchronize() override;
};

void CudaDevice::synchronize() {
    CUDA_CHECK(cuEventRecord(_sync_event, _stream));
    CUDA_CHECK(cuEventSynchronize(_sync_event));
}

CudaDevice::CudaDevice(Context *context, uint32_t device_id) : Device{context} {

    static std::once_flag flag;
    std::call_once(flag, cuInit, 0);

    int count = 0;
    CUDA_CHECK(cuDeviceGetCount(&count));
    LUISA_ERROR_IF_NOT(device_id < count, "Invalid CUDA device index ", device_id, ": max available index is ", count - 1, ".");

    CUDA_CHECK(cuDeviceGet(&_handle, device_id));
    CUDA_CHECK(cuCtxCreate(&_ctx, 0, _handle));

    char buffer[1024];
    CUDA_CHECK(cuDeviceGetName(buffer, 1024, _handle));
    auto major = 0;
    auto minor = 0;
    CUDA_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _handle));
    CUDA_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _handle));
    _compute_capability = static_cast<uint>(major * 10 + minor);
    LUISA_INFO("Created CUDA device #", device_id, ", description: name = ", buffer, ", arch = sm_", _compute_capability, ".");

    CUDA_CHECK(cuStreamCreate(&_stream, 0));
    CUDA_CHECK(cuEventCreate(&_sync_event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
    
    _dispatch_thread = std::thread{[this, device_id] {
        
        CUdevice device;
        CUcontext ctx;
        CUDA_CHECK(cuDeviceGet(&device, device_id));
        CUDA_CHECK(cuCtxCreate(&ctx, 0, device));
        while (!_stop_signal.load()) {
            using namespace std::chrono_literals;
            std::unique_lock lock{_dispatch_mutex};
            _dispatch_cv.wait_for(lock, 5ms, [this] { return !_dispatch_queue.empty(); });
            if (!_dispatch_queue.empty()) {
                auto dispatch = std::move(_dispatch_queue.front());
                _dispatch_queue.pop();
                dispatch->wait();
            }
        }
        CUDA_CHECK(cuCtxDestroy(ctx));
    }};
}

CudaDevice::~CudaDevice() noexcept {
    _stop_signal = true;
    _dispatch_thread.join();
    CUDA_CHECK(cuStreamDestroy(_stream));
    for (auto module : _modules) { CUDA_CHECK(cuModuleUnload(module)); }
    CUDA_CHECK(cuCtxDestroy(_ctx));
}

std::unique_ptr<Kernel> CudaDevice::_compile_kernel(const Function &function) {
    
    std::ostringstream os;
    CudaCodegen codegen{os};
    codegen.emit(function);
    auto src = os.str();
    LUISA_INFO("Generated source:\n", src);
    
    auto digest = SHA1{src}.digest();
    auto iter = _kernel_cache.find(digest);
    
    if (iter == _kernel_cache.end()) {
        
        LUISA_INFO("No compilation cache found for kernel \"", function.name(), "\", compiling from source...");
        
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
        NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), serialize(function.name(), ".cu").c_str(), header_sources.size(), header_sources.data(), header_names.data()));// includeNames
        
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
        NVRTC_CHECK(nvrtcCompileProgram(prog, sizeof(opts) / sizeof(const char *), opts));// options
        
        size_t log_size;
        NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
        if (log_size > 1u) {
            std::string log;
            log.resize(log_size - 1u);
            NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
            LUISA_INFO("Compile log: ", log);
        }
        
        size_t ptx_size;
        NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
        std::string ptx;
        ptx.resize(ptx_size - 1u);
        NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
        NVRTC_CHECK(nvrtcDestroyProgram(&prog));
        
        jitify::detail::ptx_remove_unused_globals(&ptx);
        LUISA_INFO("Generated PTX:\n", ptx);
        
        CUmodule module;
        CUfunction kernel;
        CUDA_CHECK(cuModuleLoadData(&module, ptx.data()));
        CUDA_CHECK(cuModuleGetFunction(&kernel, module, function.name().c_str()));
        _modules.emplace_back(module);
        
        iter = _kernel_cache.emplace(digest, kernel).first;
    } else {
        LUISA_INFO("Cache hit for kernel \"", function.name(), "\", compilation skipped.");
    }
    
    return std::make_unique<CudaKernel>(iter->second, ArgumentEncoder{function.arguments()});
}

std::shared_ptr<Buffer> CudaDevice::_allocate_buffer(size_t size) {
    CUdeviceptr buffer;
    CUDA_CHECK(cuMemAlloc(&buffer, size));
    return std::make_shared<CudaBuffer>(buffer, size);
}

void CudaDevice::_launch(const std::function<void(Dispatcher &)> &dispatch) {
    auto dispatcher = std::make_unique<CudaDispatcher>(_stream);
    dispatch(*dispatcher);
    dispatcher->commit();
    {
        std::lock_guard lock{_dispatch_mutex};
        _dispatch_queue.push(std::move(dispatcher));
    }
    _dispatch_cv.notify_one();
}

}// namespace luisa::cuda

LUISA_EXPORT_DEVICE_CREATOR(luisa::cuda::CudaDevice)
