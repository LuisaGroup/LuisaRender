//
// Created by Mike on 8/27/2020.
//

#include "cuda_codegen.h"

namespace luisa::cuda {

using namespace luisa::compute;
using namespace luisa::compute::dsl;

void CudaCodegen::emit(const Function &f) {

    _os << "#define GLM_FORCE_CUDA\n"
           "#define GLM_FORCE_CXX03\n"
           "\n"
           "#include <cmath>\n"
           "#include <cstdint>\n"
           "\n"
           "using uint = uint32_t;\n"
           "\n"
           "template<typename T, uint N> using array = T[N];\n";
    CppCodegen::emit(f);
}

void CudaCodegen::_emit_function_body(const Function &f) {

    for (auto &&v : f.builtin_variables()) {
        _emit_indent();
        if (v.is_thread_id()) {
            _os << "auto tid = static_cast<uint>(blockIdx.x * blockDim.x + threadIdx.x);\n";
        } else if (v.is_thread_xy()) {
            _os << "auto txy = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);\n";
        }
    }
    CppCodegen::_emit_function_body(f);
}

void CudaCodegen::_emit_function_decl(const Function &f) {
    _os << "extern \"C\" __global__ ";
    CppCodegen::_emit_function_decl(f);
}

}// namespace luisa::cuda
