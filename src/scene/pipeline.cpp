//
// Created by Mike on 2021/12/15.
//

#include <scene/pipeline.h>

namespace luisa::render {

inline Pipeline::Pipeline(Device &device) noexcept
    : _device{device},
      _bindless_array{device.create_bindless_array(bindless_array_capacity)} {}

}// namespace luisa::render
