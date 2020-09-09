//
// Created by Mike Smith on 2020/8/13.
//

#pragma once

#include <cstdint>
#include <core/data_types.h>
#include <core/logging.h>
#include <compute/buffer.h>

namespace luisa::compute {

enum struct PixelFormat : uint32_t {
    R8U, RG8U, RGBA8U,
    R32F, RG32F, RGBA32F,
};

namespace detail {

template<typename T>
struct PixelFormatImpl {
    
    template<typename U>
    static constexpr auto always_false = false;
    
    static_assert(always_false<T>, "Unsupported type for pixel format.");
};

#define MAKE_PIXEL_FORMAT_OF_TYPE(Type, f)          \
template<>                                          \
struct PixelFormatImpl<Type> {                      \
    static constexpr auto format = PixelFormat::f;  \
};                                                  \

MAKE_PIXEL_FORMAT_OF_TYPE(uchar, R8U)
MAKE_PIXEL_FORMAT_OF_TYPE(uchar2, RG8U)
MAKE_PIXEL_FORMAT_OF_TYPE(uchar4, RGBA8U)
MAKE_PIXEL_FORMAT_OF_TYPE(float, R32F)
MAKE_PIXEL_FORMAT_OF_TYPE(float2, RG32F)
MAKE_PIXEL_FORMAT_OF_TYPE(float4, RGBA32F)

#undef MAKE_PIXEL_FORMAT_OF_TYPE

}

template<typename T>
constexpr auto pixel_format = detail::PixelFormatImpl<T>::format;

class TextureView;

class Texture : public Noncopyable, public std::enable_shared_from_this<Texture> {

protected:
    uint32_t _width;
    uint32_t _height;
    PixelFormat _format;

public:
    Texture(uint32_t width, uint32_t height, PixelFormat format) noexcept
        : _width{width}, _height{height}, _format{format} {}
    virtual ~Texture() noexcept = default;
    [[nodiscard]] uint32_t width() const noexcept { return _width; }
    [[nodiscard]] uint32_t height() const noexcept { return _height; }
    [[nodiscard]] PixelFormat format() const noexcept { return _format; }
    virtual void clear_cache() = 0;
    
    virtual void copy_from(Dispatcher &dispatcher, Buffer *buffer, size_t offset) = 0;
    virtual void copy_to(Dispatcher &dispatcher, Buffer *buffer, size_t offset) = 0;
    virtual void copy_to(Dispatcher &dispatcher, Texture *texture) = 0;
    virtual void copy_from(Dispatcher &dispatcher, const void *data) = 0;
    virtual void copy_to(Dispatcher &dispatcher, void *data) = 0;
    
    [[nodiscard]] TextureView view() noexcept;
};

class TextureView {

private:
    std::shared_ptr<Texture> _texture{nullptr};

public:
    TextureView() noexcept = default;
    explicit TextureView(std::shared_ptr<Texture> texture) noexcept : _texture{std::move(texture)} {}
    [[nodiscard]] Texture *texture() const noexcept { return _texture.get(); }
    
    [[nodiscard]] auto copy_from(const void *data) { return [this, data](Dispatcher &d) { _texture->copy_from(d, data); }; }
    [[nodiscard]] auto copy_to(void *data) { return [this, data](Dispatcher &d) { _texture->copy_to(d, data); }; }
    
    template<typename T>
    [[nodiscard]] auto copy_from(const BufferView<T> &buffer) {
        LUISA_WARNING_IF_NOT(pixel_format<T> == _texture->format(), "Texture pixel format and buffer type mismatch.");
        return [this, buffer](Dispatcher &d) { _copy_from(d, buffer.buffer(), buffer.byte_offset()); };
    }
    
    template<typename T>
    [[nodiscard]] auto copy_to(const BufferView<T> &buffer) {
        LUISA_WARNING_IF_NOT(pixel_format<T> == _texture->format(), "Texture pixel format and buffer type mismatch.");
        return [this, buffer](Dispatcher &d) { _copy_to(d, buffer.buffer(), buffer.byte_offset()); };
    }
    
    [[nodiscard]] auto copy_to(const TextureView &tv) { return [this, &tv](Dispatcher &d) { _texture->copy_to(d, tv.texture()); }; }
    
    [[nodiscard]] uint32_t width() const noexcept { return _texture->width(); }
    [[nodiscard]] uint32_t height() const noexcept { return _texture->height(); }
    [[nodiscard]] PixelFormat format() const noexcept { return _texture->format(); }
    
    void clear_cache() { _texture->clear_cache(); }
    
    [[nodiscard]] uint32_t channels() const noexcept {
        if (_texture->format() == PixelFormat::R8U || _texture->format() == PixelFormat::R32F) { return 1u; }
        if (_texture->format() == PixelFormat::RG8U || _texture->format() == PixelFormat::RG32F) { return 2u; }
        return 4u;
    }
    
    [[nodiscard]] uint32_t pixel_byte_size() const noexcept {
        if (_texture->format() == PixelFormat::R8U ||
            _texture->format() == PixelFormat::RG8U ||
            _texture->format() == PixelFormat::RGBA8U) {
            return channels();
        }
        return sizeof(float) * channels();
    }
    
    [[nodiscard]] uint32_t pitch_byte_size() const noexcept { return pixel_byte_size() * width(); }
    [[nodiscard]] uint32_t byte_size() const noexcept { return pitch_byte_size() * height(); }
    [[nodiscard]] uint32_t pixel_count() const noexcept { return width() * height(); }
    
    // For DSL
    template<typename UV>
    [[nodiscard]] auto read(UV &&uv) const noexcept {
        using namespace luisa::compute::dsl;
        Expr uv_expr{std::forward<UV>(uv)};
        auto tex = Variable::make_texture_argument(_texture);
        Function::current().mark_texture_read(_texture.get());
        return Expr<float4>{Variable::make_temporary(nullptr, std::make_unique<TextureExpr>(TextureOp::READ, tex, uv_expr.variable()))};
    }
    
    template<typename UV>
    [[nodiscard]] auto sample(UV &&uv) const noexcept {
        using namespace luisa::compute::dsl;
        Expr uv_expr{std::forward<UV>(uv)};
        auto tex = Variable::make_texture_argument(_texture);
        Function::current().mark_texture_sample(_texture.get());
        return Expr<float4>{Variable::make_temporary(nullptr, std::make_unique<TextureExpr>(TextureOp::SAMPLE, tex, uv_expr.variable()))};
    }
    
    template<typename UV, typename Value>
    [[nodiscard]] auto write(UV &&uv, Value &&value) const noexcept {
        using namespace luisa::compute::dsl;
        Expr uv_expr{std::forward<UV>(uv)};
        Expr value_expr{std::forward<Value>(value)};
        auto tex = Variable::make_texture_argument(_texture);
        Function::current().mark_texture_write(_texture.get());
        return Expr<float4>{Variable::make_temporary(nullptr, std::make_unique<TextureExpr>(TextureOp::SAMPLE, tex, uv_expr.variable(), value_expr.variable()))};
    }
};

TextureView Texture::view() noexcept {
    return TextureView{shared_from_this()};
}

}
