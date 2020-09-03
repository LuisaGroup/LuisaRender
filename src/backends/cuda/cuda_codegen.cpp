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
           "#include <math_util.h>\n"
           "#include <texture_util.h>\n"
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
            _os << "auto txy = luisa::make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);\n";
        }
    }
    CppCodegen::_emit_function_body(f);
}

void CudaCodegen::_emit_function_decl(const Function &f) {
    _os << "extern \"C\" __global__ void " << f.name() << "(const Argument arg) ";
}

void CudaCodegen::_emit_function_call(const std::string &name) {
    if (name == "threadgroup_barrier") {
        _os << "__syncthreads";
    } else {
        _os << "luisa::";
        CppCodegen::_emit_function_call(name);
    }
}

void CudaCodegen::_emit_argument_member_decl(Variable v) {
    if (v.type()->type == TypeCatalog::TEXTURE) {
        LUISA_ERROR_IF_NOT(v.is_texture_argument(), "Textures are only allowed to be arguments.");
        auto format = v.texture()->format();
        _emit_indent();
        switch (format) {
            case PixelFormat::R8U:
                _os << "luisa::Tex2D<uint8_t> ";
                break;
            case PixelFormat::RG8U:
                _os << "luisa::Tex2D<luisa::uchar2> ";
                break;
            case PixelFormat::RGBA8U:
                _os << "luisa::Tex2D<luisa::uchar4> ";
                break;
            case PixelFormat::R32F:
                _os << "luisa::Tex2D<float> ";
                break;
            case PixelFormat::RG32F:
                _os << "luisa::Tex2D<luisa::float2> ";
                break;
            case PixelFormat::RGBA32F:
                _os << "luisa::Tex2D<luisa::float4> ";
                break;
            default:
                break;
        }
        _os << "v" << v.uid();
    } else {
        CppCodegen::_emit_argument_member_decl(v);
    }
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

void CudaCodegen::_emit_variable_decl(Variable v) {
    if (v.is_threadgroup()) { _os << "__shared__ "; }
    _emit_type(v.type());
    _os << " v" << v.uid();
}

}// namespace luisa::cuda
