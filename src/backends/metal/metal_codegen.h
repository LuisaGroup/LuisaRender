//
// Created by Mike Smith on 2020/7/31.
//

#pragma once

#include <compute/device.h>
#include <compute/codegen.h>

namespace luisa::metal {

class MetalCodegen : public dsl::CppCodegen {

public:
    MetalCodegen(std::ostream &os, Device *device) noexcept : dsl::CppCodegen{os, device} {}
    
};

}
