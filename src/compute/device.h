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

#include <core/concepts.h>
#include <core/context.h>

#include <compute/kernel.h>

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
    Context *_context;
    virtual void _launch_async(std::function<void(KernelDispatcher &)> dispatch, std::function<void()> callback) = 0;

public:
    explicit Device(Context *context) noexcept;
    virtual ~Device() noexcept;
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
    [[nodiscard]] const Context &context() const noexcept { return *_context; }
    
    static std::unique_ptr<Device> create(Context *context, std::string_view name);
};

using DeviceCreator = Device *(Context *);

#define LUISA_EXPORT_DEVICE_CREATOR(DeviceClass)  \
    LUISA_DLL_EXPORT ::luisa::Device *create(::luisa::Context *context) { return new DeviceClass{context}; }

}
