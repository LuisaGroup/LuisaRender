//
// Created by Mike Smith on 2022/3/8.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::render {

using compute::Buffer;
using compute::Expr;
using compute::Float;
using compute::UInt;

template<typename Buffer, typename Op, typename Index>
    requires compute::is_integral_expr_v<Index> &&
        compute::is_integral_v<compute::buffer_element_t<compute::expr_value_t<Buffer>>>
inline auto atomic_update(Buffer &&buffer, Index &&index, Op &&op) noexcept {
    using T = compute::buffer_element_t<compute::expr_value_t<Buffer>>;
    auto old_value = compute::def(static_cast<T>(0));
    auto i = compute::def(std::forward<Index>(index));
    compute::loop([&] {
        old_value = buffer.read(i);
        auto new_value = as<T>(op(old_value));
        auto curr_value = buffer.atomic(i).compare_exchange(old_value, new_value);
        compute::if_(curr_value == old_value, compute::break_);
    });
    return old_value;
}

template<typename Buffer, typename Index>
Float atomic_float_add(Buffer &&buffer, Index &&index, Expr<float> x) noexcept {
    auto old = atomic_update(
        std::forward<Buffer>(buffer),
        std::forward<Index>(index),
        [x](Expr<uint> old) { return compute::as<float>(old) + x; });
    return compute::as<float>(old);
}

}// namespace luisa::render
