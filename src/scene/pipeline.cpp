//
// Created by Mike on 2021/12/15.
//

#include <scene/pipeline.h>

namespace luisa::render {

template<typename T, size_t buffer_id_shift, size_t buffer_element_alignment>
inline void Pipeline::BufferArena<T, buffer_id_shift, buffer_element_alignment>::_create_buffer() noexcept {
    _buffer = _pipeline.create<Buffer<T>>(buffer_capacity);
    _buffer_id = _pipeline.register_bindless(_buffer->view());
    _buffer_offset = 0u;
}

template<typename T, size_t buffer_id_shift, size_t buffer_element_alignment>
std::pair<BufferView<T>, uint> Pipeline::BufferArena<T, buffer_id_shift, buffer_element_alignment>::allocate(size_t n) noexcept {
    if (n > buffer_capacity) {// too big, will not use the arena
        auto buffer =_pipeline.create<Buffer<T>>(n);
        auto buffer_id = _pipeline.register_bindless(buffer->view());
        return std::make_pair(buffer->view(), buffer_id << buffer_id_shift);
    }
    static constexpr auto a = buffer_element_alignment;
    if (_buffer == nullptr || _buffer_offset + n > buffer_capacity) {
        _create_buffer();
    }
    auto view = _buffer->view(_buffer_offset, n);
    auto id_and_offset = (_buffer_id << buffer_id_shift) |
                         (_buffer_offset / buffer_element_alignment);
    _buffer_offset = (_buffer_offset + n + a - 1u) / a * a;
    return std::make_pair(view, id_and_offset);
}

inline Pipeline::Pipeline(Device &device) noexcept
    : _device{device},
      _bindless_array{device.create_bindless_array(bindless_array_capacity)},
      _position_buffer_arena{*this},
      _attribute_buffer_arena{*this},
      _triangle_buffer_arena{*this},
      _area_cdf_buffer_arena{*this},
      _transform_buffer_arena{*this},
      _instance_buffer_arena{*this} {}

Pipeline::~Pipeline() noexcept = default;

}// namespace luisa::render
