//
// Created by Mike on 8/27/2020.
//

#include "cuda_codegen.h"

namespace luisa::cuda {

using namespace luisa::compute;
using namespace luisa::compute::dsl;

void CudaCodegen::emit(const Function &f) {

    _os << "#include <cmath>\n"
           "#include <cstdint>\n"
           "\n"
           "#include <math_helpers.h>\n"
           "\n"
           "using luisa::uchar;\n"
           "using luisa::ushort;\n"
           "using luisa::uint;\n"
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
    _os << "extern \"C\" __global__ void " << f.name() << "(const Argument arg) ";
}

void CudaCodegen::_emit_function_call(const std::string &name) {
    _os << "luisa::math::";
    CppCodegen::_emit_function_call(name);
}

void CudaCodegen::_emit_type(const TypeDesc *desc) {
    
    switch (desc->type) {
        case TypeCatalog::VECTOR2:
            _os << "luisa::";
            _emit_type(desc->element_type);
            _os << 2;
            break;
        case TypeCatalog::VECTOR3:
            _os << "luisa::";
            _emit_type(desc->element_type);
            _os << 3;
            break;
        case TypeCatalog::VECTOR4:
            _os << "luisa::";
            _emit_type(desc->element_type);
            _os << 4;
            break;
        case TypeCatalog::VECTOR3_PACKED:
            _os << "luisa::packed_";
            _emit_type(desc->element_type);
            _os << 3;
            break;
        case TypeCatalog::MATRIX3:
            _os << "luisa::float3x3";
            break;
        case TypeCatalog::MATRIX4:
            _os << "luisa::float4x4";
            break;
        default:
            CppCodegen::_emit_type(desc);
            break;
    }
}

}// namespace luisa::cuda
