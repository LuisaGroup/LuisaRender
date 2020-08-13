//
// Created by Mike Smith on 2020/8/14.
//

#pragma once

#include <cstdint>

namespace luisa::compute {

enum struct AccessMode : uint32_t {
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE,
    SAMPLE
};

}
