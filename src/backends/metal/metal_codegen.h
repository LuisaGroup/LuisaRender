//
// Created by Mike Smith on 2020/7/31.
//

#pragma once

#include <compute/device.h>
#include <compute/codegen.h>

namespace luisa::metal {

class MetalCodegen : public compute::dsl::CppCodegen {

protected:
    void _emit_function_decl(const compute::dsl::Function &f) override;
    void _emit_type(const compute::dsl::TypeDesc *desc) override;
    void _emit_function_call(const std::string &name) override;

public:
    MetalCodegen(std::ostream &os, Device *device) noexcept : compute::dsl::CppCodegen{os, device} {}
    void emit(const compute::dsl::Function &f) override;
};

}
