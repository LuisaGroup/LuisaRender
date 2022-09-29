//
// Created by Mike Smith on 2022/2/12.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

using compute::Expr;
using compute::UInt;
using compute::UInt2;
using compute::UInt3;
using compute::UInt4;
using compute::Float;

[[nodiscard]] UInt xxhash32(Expr<uint> p) noexcept;
[[nodiscard]] UInt xxhash32(Expr<uint2> p) noexcept;
[[nodiscard]] UInt xxhash32(Expr<uint3> p) noexcept;
[[nodiscard]] UInt xxhash32(Expr<uint4> p) noexcept;

[[nodiscard]] UInt pcg(Expr<uint> v) noexcept;
[[nodiscard]] UInt2 pcg2d(Expr<uint2> v_in) noexcept;
[[nodiscard]] UInt3 pcg3d(Expr<uint3> v_in) noexcept;
[[nodiscard]] UInt4 pcg4d(Expr<uint4> v_in) noexcept;

[[nodiscard]] Float lcg(UInt &state) noexcept;

}// namespace luisa::render
