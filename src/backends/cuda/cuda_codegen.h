//
// Created by Mike on 8/27/2020.
//

#pragma once

#include <compute/codegen.h>

namespace luisa::cuda {

using luisa::compute::dsl::Function;
using luisa::compute::dsl::CppCodegen;

class CudaCodegen : public CppCodegen {

private:


public:
    explicit CudaCodegen(std::ostream &os) noexcept : CppCodegen{os} {}
    void emit(const Function &f) override;
};

}
