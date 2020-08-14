//
// Created by Mike Smith on 2020/8/9.
//

#pragma once

#include <core/concepts.h>

namespace luisa::compute {

struct Dispatcher : Noncopyable {
    virtual void synchronize() = 0;
};

}
