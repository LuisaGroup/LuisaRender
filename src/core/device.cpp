//
// Created by Mike Smith on 2020/2/29.
//

#include "device.h"

namespace luisa {

void Device::_register_creator(std::string_view name, std::function<std::unique_ptr<Device>()> creator) {
    _device_creators[name] = std::move(creator);
}

Device::~Device() noexcept { synchronize(); }

std::unique_ptr<Device> Device::create(std::string_view name) {
    LUISA_INFO("Creating backend: ", name);
    return _device_creators.at(name)();
}

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

Device::Device() noexcept
    : _command_queue_size{16u}, _working_command_count{0u} {}

}