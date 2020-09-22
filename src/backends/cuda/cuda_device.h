//
// Created by Mike on 9/22/2020.
//

#pragma once

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
    std::mutex _kernel_cache_mutex;
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
        const BufferView<TriangleHandle> &indices,
        const std::vector<MeshHandle> &meshes,
        const BufferView<uint> &instances,
        const BufferView<float4x4> &transforms,
        bool is_static) override;
};

}
