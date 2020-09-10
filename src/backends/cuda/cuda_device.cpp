//
// Created by Mike on 8/27/2020.
//

#include <cuda.h>
#include <nvrtc.h>

#include <jitify/jitify.hpp>

#include <core/hash.h>
#include <compute/device.h>
#include <compute/kernel.h>
#include <compute/acceleration.h>

#include "cuda_buffer.h"
#include "cuda_texture.h"
#include "cuda_codegen.h"
#include "cuda_dispatcher.h"
#include "cuda_jit_headers.h"
#include "cuda_kernel.h"

namespace luisa::cuda {

using namespace luisa::compute;

class CudaDevice : public Device {

private:
    CUdevice _handle{};
    CUcontext _ctx{};
    CUstream _dispatch_stream{};
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
    std::shared_ptr<Texture> _allocate_texture(uint32_t width, uint32_t height, compute::PixelFormat format) override;
    std::shared_ptr<Kernel> _compile_kernel(const compute::dsl::Function &function) override;

    void _launch(const std::function<void(Dispatcher &)> &work) override;

public:
    explicit CudaDevice(Context *context, uint32_t device_id);
    ~CudaDevice() noexcept override;
    void synchronize() override;
    std::unique_ptr<Acceleration> build_acceleration(
        const BufferView<float3> &positions,
        const BufferView<packed_uint3> &indices,
        const std::vector<packed_uint3> &meshes,
        const BufferView<uint> &instances,
        const BufferView<float4x4> &transforms,
        bool is_static) override;
};

void CudaDevice::synchronize() {
    CUDA_CHECK(cuEventRecord(_sync_event, _dispatch_stream));
    CUDA_CHECK(cuEventSynchronize(_sync_event));
}

CudaDevice::CudaDevice(Context *context, uint32_t device_id) : Device{context} {

    static std::once_flag flag;
    std::call_once(flag, cuInit, 0);

    int count = 0;
    CUDA_CHECK(cuDeviceGetCount(&count));
    LUISA_ERROR_IF_NOT(device_id < count, "Invalid CUDA device index ", device_id, ": max available index is ", count - 1, ".");

    CUDA_CHECK(cuDeviceGet(&_handle, device_id));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&_ctx, _handle));
    CUDA_CHECK(cuCtxSetCurrent(_ctx));

    char buffer[1024];
    CUDA_CHECK(cuDeviceGetName(buffer, 1024, _handle));
    auto major = 0;
    auto minor = 0;
    CUDA_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _handle));
    CUDA_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _handle));
    _compute_capability = static_cast<uint>(major * 10 + minor);
    LUISA_INFO("Created CUDA device #", device_id, ", description: name = ", buffer, ", arch = sm_", _compute_capability, ".");

    CUDA_CHECK(cuStreamCreate(&_dispatch_stream, 0));
    CUDA_CHECK(cuEventCreate(&_sync_event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
    
    _dispatch_thread = std::thread{[this, device = _handle] {
        CUcontext ctx;
        CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
        CUDA_CHECK(cuCtxSetCurrent(ctx));
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
        CUDA_CHECK(cuDevicePrimaryCtxRelease(device));
    }};
}

CudaDevice::~CudaDevice() noexcept {
    _stop_signal = true;
    _dispatch_thread.join();
    CUDA_CHECK(cuStreamDestroy(_dispatch_stream));
    CUDA_CHECK(cuEventDestroy(_sync_event));
    for (auto module : _modules) { CUDA_CHECK(cuModuleUnload(module)); }
    CUDA_CHECK(cuDevicePrimaryCtxRelease(_handle));
}

std::shared_ptr<Kernel> CudaDevice::_compile_kernel(const Function &function) {
    
    std::ostringstream os;
    CudaCodegen codegen{os};
    codegen.emit(function);
    auto src = os.str();
    LUISA_INFO("Generated source:\n", src);
    
    auto digest = SHA1{src}.digest();
    auto iter = _kernel_cache.find(digest);
    
    if (iter == _kernel_cache.end()) {
    
        std::ostringstream ss;
        for (auto d : digest) { ss << std::setfill('0') << std::setw(8) << std::hex << std::uppercase << d; }
        auto digest_str = ss.str();
        auto cache_file_path = _context->cache_path(digest_str.append(".ptx"));
        LUISA_INFO("No cache found for kernel \"", function.name(), "\" in memory, searching on disk: ", cache_file_path);
        
        if (std::filesystem::exists(cache_file_path)) {
            LUISA_INFO("Cache hit for kernel \"", function.name(), "\" on disk, compilation skipped.");
            auto ptx = text_file_contents(cache_file_path);
            CUmodule module;
            CUfunction kernel;
            CUDA_CHECK(cuModuleLoadData(&module, ptx.data()));
            CUDA_CHECK(cuModuleGetFunction(&kernel, module, function.name().c_str()));
            _modules.emplace_back(module);
            iter = _kernel_cache.emplace(digest, kernel).first;
        } else {
            
            LUISA_INFO("No cache found for kernel \"", function.name(), "\" on disk, compiling from source...");
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
            nvrtcCompileProgram(prog, sizeof(opts) / sizeof(const char *), opts);// options

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
            //        LUISA_INFO("Generated PTX:\n", ptx);

            CUmodule module;
            CUfunction kernel;
            CUDA_CHECK(cuModuleLoadData(&module, ptx.data()));
            CUDA_CHECK(cuModuleGetFunction(&kernel, module, function.name().c_str()));
            _modules.emplace_back(module);
            
            LUISA_INFO("Writing cache for compiled kernel \"", function.name(), "\" to disk: ", cache_file_path);
            std::ofstream ptx_file{cache_file_path};
            ptx_file << ptx;
            
            iter = _kernel_cache.emplace(digest, kernel).first;
        }
    } else {
        LUISA_INFO("Cache hit for kernel \"", function.name(), "\" in memory, compilation skipped.");
    }
    
    std::vector<Kernel::Resource> resources;
    std::vector<Kernel::Uniform> uniforms;
    size_t uniform_offset = 0u;
    for (auto &&arg : function.arguments()) {
        if (arg->is_buffer_argument()) {
            Kernel::Resource r;
            r.buffer = arg->buffer()->shared_from_this();
            resources.emplace_back(std::move(r));
        } else if (arg->is_texture_argument()) {
            Kernel::Resource r;
            r.texture = arg->texture()->shared_from_this();
            resources.emplace_back(std::move(r));
        } else if (arg->is_uniform_argument()) {
            auto alignment = arg->type()->alignment;
            uniform_offset = (uniform_offset + alignment - 1u) / alignment * alignment;
            Kernel::Uniform uniform;
            uniform.offset = uniform_offset;
            uniform.binding = arg->uniform_data();
            uniform.binding_size = arg->type()->size;
            uniforms.emplace_back(std::move(uniform));
            uniform_offset += arg->type()->size;
        } else if (arg->is_immutable_argument()) {
            auto alignment = arg->type()->alignment;
            uniform_offset = (uniform_offset + alignment - 1u) / alignment * alignment;
            Kernel::Uniform uniform;
            uniform.immutable = arg->immutable_data();
            uniform.offset = uniform_offset;
            uniforms.emplace_back(std::move(uniform));
            uniform_offset += arg->type()->size;
        }
    }
    return std::make_shared<CudaKernel>(iter->second, std::move(resources), std::move(uniforms));
}

std::shared_ptr<Buffer> CudaDevice::_allocate_buffer(size_t size) {
    CUdeviceptr buffer;
    CUDA_CHECK(cuMemAlloc(&buffer, size));
    return std::make_shared<CudaBuffer>(buffer, size);
}

void CudaDevice::_launch(const std::function<void(Dispatcher &)> &work) {
    auto dispatcher = std::make_unique<CudaDispatcher>(_dispatch_stream);
    (*dispatcher)(work);
    dispatcher->commit();
    {
        std::lock_guard lock{_dispatch_mutex};
        _dispatch_queue.push(std::move(dispatcher));
    }
    _dispatch_cv.notify_one();
}

std::shared_ptr<Texture> CudaDevice::_allocate_texture(uint32_t width, uint32_t height, compute::PixelFormat format) {
    
    // Create array
    CUDA_ARRAY_DESCRIPTOR array_desc{};
    array_desc.Width = width;
    array_desc.Height = height;
    switch (format) {
        case compute::PixelFormat::R8U:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 1;
            break;
        case compute::PixelFormat::RG8U:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 2;
            break;
        case compute::PixelFormat::RGBA8U:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 4;
            break;
        case compute::PixelFormat::R32F:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 1;
            break;
        case compute::PixelFormat::RG32F:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 2;
            break;
        case compute::PixelFormat::RGBA32F:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 4;
            break;
        default:
            break;
    }
    CUarray array;
    CUDA_CHECK(cuArrayCreate(&array, &array_desc));
    
    // Create texture & surface
    CUDA_RESOURCE_DESC res_desc{};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = array;
    res_desc.flags = 0;
    
    CUDA_TEXTURE_DESC tex_desc{};
    tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    tex_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    tex_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    tex_desc.filterMode =  CU_TR_FILTER_MODE_LINEAR;
    tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    
    CUDA_RESOURCE_VIEW_DESC res_view_desc{};
    switch (format) {
        case PixelFormat::R8U:
            res_view_desc.format = CU_RES_VIEW_FORMAT_UINT_1X8;
            break;
        case PixelFormat::RG8U:
            res_view_desc.format = CU_RES_VIEW_FORMAT_UINT_2X8;
            break;
        case PixelFormat::RGBA8U:
            res_view_desc.format = CU_RES_VIEW_FORMAT_UINT_4X8;
            break;
        case PixelFormat::R32F:
            res_view_desc.format = CU_RES_VIEW_FORMAT_FLOAT_1X32;
            break;
        case PixelFormat::RG32F:
            res_view_desc.format = CU_RES_VIEW_FORMAT_FLOAT_2X32;
            break;
        case PixelFormat::RGBA32F:
            res_view_desc.format = CU_RES_VIEW_FORMAT_FLOAT_4X32;
            break;
        default:
            break;
    }
    res_view_desc.width = width;
    res_view_desc.height = height;
    
    CUtexObject texture;
    CUsurfObject surface;
    CUDA_CHECK(cuTexObjectCreate(&texture, &res_desc, &tex_desc, &res_view_desc));
    CUDA_CHECK(cuSurfObjectCreate(&surface, &res_desc));
    
    return std::make_shared<CudaTexture>(array, texture, surface, width, height, format);
}

std::unique_ptr<Acceleration> CudaDevice::build_acceleration(
    const BufferView<float3> &positions,
    const BufferView<packed_uint3> &indices,
    const std::vector<packed_uint3> &meshes,
    const BufferView<uint> &instances,
    const BufferView<float4x4> &transforms,
    bool is_static) {
    
    // TODO
    LUISA_ERROR("Not implemented!");
}

}// namespace luisa::cuda

LUISA_EXPORT_DEVICE_CREATOR(luisa::cuda::CudaDevice)
