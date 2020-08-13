//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <compute/v2/buffer.h>

namespace luisa::compute {

class ArgumentBinding {

private:
    Buffer *_buffer{nullptr};
    void *_data{nullptr};
    size_t _offset{0u};
    size_t _size{0u};

};

class Kernel {

};

}
