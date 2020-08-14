//
// Created by Mike Smith on 2020/8/13.
//

#pragma once

#include <cstdint>

namespace luisa::compute {

enum struct TextureAccess : uint32_t {
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE,
    SAMPLE
};

class Texture {
    
};

}
