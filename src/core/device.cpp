//
// Created by Mike Smith on 2020/2/29.
//

#include "device.h"

namespace luisa {

Device::~Device() noexcept { synchronize(); }

void Device::launch(std::function<void(KernelDispatcher &)> dispatch) {
    launch_async(std::move(dispatch));
    synchronize();
}

void Device::synchronize() {
    std::unique_lock lock{_mutex};
    _cv.wait(lock, [this] { return _working_command_count == 0u; });
}

void Device::launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) {
    
    // wait for command queue
    std::unique_lock lock{_mutex};
    _cv.wait(lock, [this]() noexcept { return _working_command_count < _command_queue_size; });
    _working_command_count++;
    lock.unlock();
    
    // launch
    _launch_async(std::move(dispatch), [this, callback = std::move(callback)] {
        std::exception_ptr exception_ptr;
        try {
            std::lock_guard lock_guard{_mutex};
            _working_command_count--;
            callback();
        } catch (...) {
            exception_ptr = std::current_exception();
        }
        _cv.notify_one();
        if (exception_ptr != nullptr) { std::rethrow_exception(exception_ptr); }
    });
}

void Device::set_command_queue_size(uint size) {
    std::lock_guard lock_guard{_mutex};
    _command_queue_size = std::max(size, 1u);
}

Device::Device(Context *context) noexcept
    : _context{context}, _command_queue_size{16u}, _working_command_count{0u} {}

std::unique_ptr<Device> Device::create(Context *context, std::string_view name) {
    auto create_device = context->load_dynamic_function<DeviceCreator>(context->runtime_path("backends") / name, name, "create");
    return std::unique_ptr<Device>{create_device(context)};
}

}
