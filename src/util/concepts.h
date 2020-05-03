//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

namespace luisa { inline namespace utility {

struct Noncopyable {
    Noncopyable() = default;
    Noncopyable(const Noncopyable &) = delete;
    Noncopyable &operator=(const Noncopyable &) = delete;
};

}}
