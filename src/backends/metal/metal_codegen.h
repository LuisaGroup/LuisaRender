//
// Created by Mike Smith on 2020/7/31.
//

#pragma once

#import <compute/device.h>
#import <compute/codegen.h>

namespace luisa::metal {

class MetalCodegen : public compute::dsl::CppCodegen {

protected:
    void _emit_function_decl(const compute::dsl::Function &f) override;
    void _emit_type(const compute::dsl::TypeDesc *desc) override;
    void _emit_builtin_function_name(const std::string &name) override;
    void _emit_variable(const compute::dsl::Variable *v) override;

public:
    explicit MetalCodegen(std::ostream &os) noexcept : compute::dsl::CppCodegen{os} {}
    void emit(const compute::dsl::Function &f) override;
};

}
