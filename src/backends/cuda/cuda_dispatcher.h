//
// Created by Mike on 9/1/2020.
//

#pragma

#include <compute/dispatcher.h>
#include <cuda.h>

namespace luisa::cuda {

using luisa::compute::Dispatcher;

class CudaDispatcher : public Dispatcher {

private:
    CUstream _handle;
    bool _finished{false};
    std::mutex _mutex;
    std::condition_variable _cv;

    void _wait() noexcept {
        std::unique_lock lock{_mutex};
        _cv.wait(lock, [this] { return _finished; });
    }

    void _notify() noexcept {
        {
            std::lock_guard lock{_mutex};
            _finished = true;
        }
        _cv.notify_one();
    }

public:
    explicit CudaDispatcher(CUstream handle) noexcept : _handle{handle} {}
    ~CudaDispatcher() noexcept override { _wait(); }

    void reset() noexcept {
        _callbacks.clear();
        _finished = false;
    }

    [[nodiscard]] CUstream handle() const noexcept { return _handle; }

    void commit() noexcept override {
        CUDA_CHECK(cuLaunchHostFunc(
            _handle, [](void *self) {
                auto dispatcher = reinterpret_cast<CudaDispatcher *>(self);
                for (auto &&cb : dispatcher->_callbacks) { cb(); }
                dispatcher->_notify();
            },
            this));
    }

    void wait() override { _wait(); }
};

}// namespace luisa::cuda
