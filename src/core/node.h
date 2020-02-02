//
// Created by Mike Smith on 2020/2/2.
//

#pragma once

#include <util/noncopyable.h>

#include "data_types.h"
#include "device.h"

namespace luisa {

class Node : public Noncopyable {

private:
    Device *_device;

public:
    explicit Node(Device *device) noexcept : _device{device} {}
    [[nodiscard]] Device &device() noexcept { return *_device; }
    
};

}
