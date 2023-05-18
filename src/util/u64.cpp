//
// Created by Mike Smith on 2023/5/18.
//

#include <core/logging.h>
#include <util/u64.h>

namespace luisa::render {

compute::UInt U64::operator%(uint rhs) const noexcept {
    LUISA_ASSERT(rhs <= 0xffffu, "U64::operator% rhs must be <= 0xffff");
    return ((hi() % rhs) * static_cast<uint>(0x1'0000'0000ull % rhs) + lo() % rhs) % rhs;
}

}// namespace luisa::render
