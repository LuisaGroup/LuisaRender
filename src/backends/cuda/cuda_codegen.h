//
// Created by Mike on 8/27/2020.
//

#pragma once

#include <compute/codegen.h>

namespace luisa::cuda {

using luisa::compute::dsl::Function;
using luisa::compute::dsl::CppCodegen;

class CudaCodegen : public CppCodegen {

protected:
    void _emit_function_body(const Function &f) override;
    void _emit_function_decl(const Function &f) override;
    void _emit_type(const compute::dsl::TypeDesc *desc) override;
    void _emit_function_call(const std::string &name) override;
    void _emit_argument_struct_decl(const Function &f) override;
    void _emit_variable(compute::dsl::Variable v) override;

public:
    explicit CudaCodegen(std::ostream &os) noexcept : CppCodegen{os} {}
    void emit(const Function &f) override;
};

}
