//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <memory>
#include <filesystem>
#include <unordered_map>
#include <functional>
#include <utility>
#include <iostream>
#include <condition_variable>
#include <mutex>

#include <util/noncopyable.h>

#include "kernel.h"

namespace luisa {
class Geometry;
struct Acceleration;
}

namespace luisa {

class Device : Noncopyable {

private:
    std::condition_variable _cv;
    std::mutex _mutex;
    uint _command_queue_size{32u};
    uint _working_command_count{0u};
    inline static std::unordered_map<std::string_view, std::function<std::unique_ptr<Device>()>> _device_creators{};

protected:
    static void _register_creator(std::string_view name, std::function<std::unique_ptr<Device>()> creator) {
        _device_creators[name] = std::move(creator);
    }
    
    virtual void _launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) = 0;

public:
    virtual ~Device() noexcept { synchronize(); }
    
    [[nodiscard]] static std::unique_ptr<Device> create(std::string_view name) {
        return _device_creators.at(name)();
    }
    
    [[nodiscard]] virtual std::unique_ptr<Kernel> create_kernel(std::string_view function_name) = 0;
    [[nodiscard]] virtual std::unique_ptr<TypelessBuffer> allocate_buffer(size_t capacity, BufferStorage storage) = 0;
    [[nodiscard]] virtual std::unique_ptr<Acceleration> create_acceleration(Geometry &geometry) = 0;
    
    template<typename T>
    [[nodiscard]] auto create_buffer(size_t element_count, BufferStorage buffer_storage) {
        return std::make_unique<Buffer<T>>(allocate_buffer(element_count * sizeof(T), buffer_storage));
    }
    
    virtual void launch(std::function<void(KernelDispatcher &)> dispatch) {
        launch_async(std::move(dispatch));
        synchronize();
    }
    
    void synchronize() {
        std::unique_lock lock{_mutex};
        _cv.wait(lock, [this] { return _working_command_count == 0u; });
    }
    
    void launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback = [] {}) {
        
        // wait for command queue
        std::unique_lock lock{_mutex};
        _cv.wait(lock, [this]() noexcept { return _working_command_count < _command_queue_size; });
        _working_command_count++;
        lock.unlock();
        
        // launch
        _launch_async(std::move(dispatch), [this, callback = std::move(callback)] {
            {
                std::lock_guard lock_guard{_mutex};
                _working_command_count--;
                callback();
            }
            _cv.notify_one();
        });
    }
    
};

#define LUISA_DEVICE_CREATOR(name)                                                                      \
        static_assert(true);                                                                            \
    private:                                                                                            \
        inline static struct _reg_helper_impl {                                                         \
            _reg_helper_impl() noexcept { Device::_register_creator(name, [] { return _create(); }); }  \
        } _reg_helper{};                                                                                \
        [[nodiscard]] static std::unique_ptr<Device> _create()

}
        