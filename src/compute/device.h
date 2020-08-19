//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <memory>

#include <core/context.h>
#include <compute/buffer.h>
#include <compute/texture.h>
#include <compute/kernel.h>
#include <compute/function.h>

namespace luisa::compute {

class Device : Noncopyable {

protected:
    Context *_context{nullptr};
    std::vector<std::unique_ptr<Buffer>> _buffers;
    
    [[nodiscard]] virtual std::unique_ptr<Buffer> _allocate_buffer(size_t size, size_t max_host_caches) = 0;
    [[nodiscard]] virtual std::unique_ptr<Texture> _allocate_texture(uint32_t width, uint32_t height, PixelFormat format, size_t max_caches) = 0;
    [[nodiscard]] virtual std::unique_ptr<Kernel> _compile_kernel(const dsl::Function &function) = 0;
    
    virtual void _launch(const std::function<void(Dispatcher &)> &dispatch) = 0;

public:
    explicit Device(Context *context) noexcept: _context{context} {}
    virtual ~Device() noexcept = default;
    
    [[nodiscard]] Context &context() const noexcept { return *_context; }
    
    template<typename Def, std::enable_if_t<std::is_invocable_v<Def>, int> = 0>
    [[nodiscard]] std::unique_ptr<Kernel> compile_kernel(std::string name, Def &&def) {
        dsl::Function function{std::move(name)};
        def();
        return _compile_kernel(function);
    }
    
    template<typename Def, std::enable_if_t<std::is_invocable_v<Def>, int> = 0>
    [[nodiscard]] std::unique_ptr<Kernel> compile_kernel(Def &&def) {
        return compile_kernel("foo", std::forward<Def>(def));
    }
    
    template<typename T>
    [[nodiscard]] BufferView<T> allocate_buffer(size_t size, size_t max_host_cache_count = 8) {
        return _buffers.emplace_back(_allocate_buffer(size * sizeof(T), max_host_cache_count))->view<T>();
    }
    
    template<typename T>
    [[nodiscard]] std::unique_ptr<Texture> allocate_texture(uint32_t width, uint32_t height, size_t max_caches = 2) {
        return allocate_texture(width, height, pixel_format<T>, max_caches);
    }
    
    [[nodiscard]] std::unique_ptr<Texture> allocate_texture(uint32_t width, uint32_t height, PixelFormat format, size_t max_caches = 2) {
        return _allocate_texture(width, height, format, max_caches);
    }
    
    template<typename Work, std::enable_if_t<std::is_invocable_v<Work, Dispatcher &>, int> = 0>
    void launch(Work &&work) {
        _launch([&work, this](Dispatcher &dispatch) { work(dispatch); });
    }
    
    void launch(Kernel &kernel, uint threads, uint tg_size = 128u) { _launch([&](Dispatcher &dispatch) { dispatch(kernel, threads, tg_size); }); }
    void launch(Kernel &kernel, uint2 threads, uint2 tg_size = make_uint2(8u, 8u)) { _launch([&](Dispatcher &dispatch) { dispatch(kernel, threads, tg_size); }); }
    void launch(Kernel &kernel, uint3 threads, uint3 tg_size) { _launch([&](Dispatcher &dispatch) { dispatch(kernel, threads, tg_size); }); }
    
    virtual void synchronize() = 0;
    
    static std::unique_ptr<Device> create(Context *context, std::string_view name);
};

using DeviceCreator = Device *(Context *);

#define LUISA_EXPORT_DEVICE_CREATOR(DeviceClass)  \
    LUISA_DLL_EXPORT ::luisa::compute::Device *create(::luisa::Context *context) { return new DeviceClass{context}; }
    
}
