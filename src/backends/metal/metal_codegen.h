//
// Created by Mike Smith on 2020/7/31.
//

#pragma once

#include <compute/device.h>
#include <compute/codegen.h>

namespace luisa::metal {

class MetalCodegen : public dsl::CppCodegen {

public:
    explicit MetalCodegen(Device *device) noexcept : dsl::CppCodegen{device} {}
    
};

}
