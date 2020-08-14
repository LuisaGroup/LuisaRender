//
// Created by Mike Smith on 2020/8/9.
//

#pragma once

#include <functional>
#include <core/concepts.h>

namespace luisa::compute {

struct Dispatcher : Noncopyable {
    virtual void commit(std::function<void()> callback) = 0;
    virtual void commit() { commit([] {}); }
    virtual void synchronize() = 0;
};

}
