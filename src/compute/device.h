//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <memory>

#include <compute/buffer.h>
#include <compute/function.h>
#include <compute/kernel.h>
#include <compute/texture.h>
#include <compute/acceleration.h>
#include <core/context.h>

namespace luisa::compute {

class Device : Noncopyable {

protected:
    Context *_context{nullptr};
    
    [[nodiscard]] virtual std::shared_ptr<Buffer> _allocate_buffer(size_t size) = 0;
    [[nodiscard]] virtual std::unique_ptr<Texture> _allocate_texture(uint32_t width, uint32_t height, PixelFormat format) = 0;
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
    [[nodiscard]] BufferView<T> allocate_buffer(size_t size) {
        return _allocate_buffer(size * sizeof(T))->view<T>();
    }
    
    template<typename T>
    [[nodiscard]] std::unique_ptr<Texture> allocate_texture(uint32_t width, uint32_t height) {
        return allocate_texture(width, height, pixel_format<T>);
    }
    
    [[nodiscard]] std::unique_ptr<Texture> allocate_texture(uint32_t width, uint32_t height, PixelFormat format) {
        return _allocate_texture(width, height, format);
    }
    
    [[nodiscard]] virtual std::unique_ptr<Acceleration> build_acceleration(
        const BufferView<float3> &positions,
        const BufferView<packed_uint3> &indices,
        const std::vector<packed_uint3> &meshes,  // (vertex offset, triangle offset, triangle count)
        const BufferView<uint> &instances,
        const BufferView<float4x4> &transforms,
        bool is_static) = 0;
    
    template<typename Work, std::enable_if_t<std::is_invocable_v<Work, Dispatcher &>, int> = 0>
    void launch(Work &&work) {
        _launch([&work, this](Dispatcher &dispatch) { dispatch(work); });
    }
    
    template<typename Work, typename Callback, std::enable_if_t<std::conjunction_v<
        std::is_invocable<Work, Dispatcher &>, std::is_invocable<Callback>>, int> = 0>
    void launch(Work &&work, Callback &&cb) {
        _launch([&work, this, &cb](Dispatcher &dispatch) { dispatch(work, cb); });
    }
    
    virtual void synchronize() = 0;
    
    [[nodiscard]] static std::unique_ptr<Device> create(Context *context, uint32_t selection_id = 0u);
};

using DeviceCreator = Device *(Context *, uint32_t);

#define LUISA_EXPORT_DEVICE_CREATOR(DeviceClass)                                                              \
    extern "C" LUISA_EXPORT ::luisa::compute::Device *create(::luisa::Context *context, uint32_t device_id) { \
        return new DeviceClass{context, device_id};                                                           \
    }
    
}// namespace luisa::compute
