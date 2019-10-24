//
// Created by Mike Smith on 2019/10/25.
//

#pragma once

struct Noncopyable {
    Noncopyable() = default;
    Noncopyable(Noncopyable &&) = delete;
    Noncopyable(const Noncopyable &) = delete;
    Noncopyable &operator=(Noncopyable &&) = delete;
    Noncopyable &operator=(const Noncopyable &) = delete;
};
