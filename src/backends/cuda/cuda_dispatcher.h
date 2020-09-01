//
// Created by Mike on 9/1/2020.
//

#pragma

#include <cuda.h>
#include <compute/dispatcher.h>

namespace luisa::cuda {

using luisa::compute::Dispatcher;

class CudaDispatcher : public Dispatcher {

private:
    CUstream _handle{nullptr};
    CUevent _event{nullptr};

    void _wait() noexcept { CUDA_CHECK(cuEventSynchronize(_event)); }

public:
    explicit CudaDispatcher(CUstream handle) noexcept : _handle{handle} {
        CUDA_CHECK(cuEventCreate(&_event, CU_EVENT_DISABLE_TIMING));
    }

    ~CudaDispatcher() noexcept override { CUDA_CHECK(cuEventDestroy(_event)); }

    [[nodiscard]] CUstream handle() const noexcept { return _handle; }

    void commit() noexcept override {
        CUDA_CHECK(cuEventRecord(_event, _handle));
    }

    void wait() override {
        _wait();
        for (auto &&cb : _callbacks) { cb(); }
    }
};

}// namespace luisa::cuda
