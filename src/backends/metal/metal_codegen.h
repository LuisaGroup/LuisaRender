//
// Created by Mike Smith on 2020/7/31.
//

#pragma once

#include <compute/device.h>
#include <compute/codegen.h>

namespace luisa::metal {

class MetalCodegen : public dsl::CppCodegen {

protected:
    void _emit_function_decl(const dsl::Function &f) override;
    void _emit_builtin_variable(BuiltinVariable tag) override;
    void _emit_argument_member_decl(dsl::Variable v) override;
    void _emit_type(const dsl::TypeDesc *desc) override;
    void _emit_function_call(const std::string &name) override;

public:
    MetalCodegen(std::ostream &os, Device *device) noexcept : dsl::CppCodegen{os, device} {}
    void emit(const dsl::Function &f) override;
};

}
