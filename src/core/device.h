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

#include <util/concepts.h>

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
    uint _command_queue_size;
    uint _working_command_count;
    inline static std::unordered_map<std::string_view, std::function<std::unique_ptr<Device>()>> _device_creators{};

protected:
    static void _register_creator(std::string_view name, std::function<std::unique_ptr<Device>()> creator);
    virtual void _launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) = 0;

public:
    Device() noexcept;
    virtual ~Device() noexcept;
    [[nodiscard]] static std::unique_ptr<Device> create(std::string_view name);
    [[nodiscard]] virtual std::unique_ptr<Kernel> load_kernel(std::string_view function_name) = 0;
    [[nodiscard]] virtual std::unique_ptr<TypelessBuffer> allocate_typeless_buffer(size_t capacity, BufferStorage storage) = 0;
    [[nodiscard]] virtual std::unique_ptr<Acceleration> build_acceleration(Geometry &geometry) = 0;
    
    template<typename T>
    [[nodiscard]] auto allocate_buffer(size_t element_count, BufferStorage buffer_storage) {
        return std::make_unique<Buffer<T>>(allocate_typeless_buffer(element_count * sizeof(T), buffer_storage));
    }
    
    virtual void launch(std::function<void(KernelDispatcher &)> dispatch);
    void synchronize();
    void launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback = [] {});
    void set_command_queue_size(uint size);
};

#define LUISA_DEVICE_CREATOR(name)                                                                      \
        static_assert(true);                                                                            \
    private:                                                                                            \
        inline static struct _reg_helper_impl {                                                         \
            _reg_helper_impl() noexcept { Device::_register_creator(name, [] { return _create(); }); }  \
        } _reg_helper{};                                                                                \
        [[nodiscard]] static std::unique_ptr<Device> _create()

}
