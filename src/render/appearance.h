//
// Created by Mike Smith on 2020/3/7.
//

#pragma once

#include <core/data_types.h>

namespace luisa::appearance {

class Info {

private:
    uint8_t _tag;
    uint8_t _index_hi;
    uint16_t _index_lo;

public:
    constexpr Info(uint tag, uint data_index)
        : _tag{static_cast<uint8_t>(tag)},
          _index_hi{static_cast<uint8_t>(data_index >> 16u)},
          _index_lo{static_cast<uint16_t>(data_index)} {}
    
    [[nodiscard]] LUISA_DEVICE_CALLABLE constexpr auto tag() const noexcept { return _tag; }
    [[nodiscard]] LUISA_DEVICE_CALLABLE constexpr auto index() const noexcept { return (static_cast<uint>(_index_hi) << 16u) | static_cast<uint>(_index_lo); }
};

static_assert(sizeof(Info) == 4ul);

}

#ifndef LUISA_DEVICE_COMPATIBLE

namespace luisa {

class Appearance {

private:


public:


};

}

#endif
