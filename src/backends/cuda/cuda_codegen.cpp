//
// Created by Mike on 8/27/2020.
//

#include <compute/texture.h>
#include <compute/buffer.h>
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
           "template<typename T, uint N>\n"
           "class array {\n"
           "private:\n"
           "    T _m[N];\n"
           "public:\n"
           "    template<typename ...Args> array(Args &&...args) noexcept : _m{args...} {}\n"
           "    [[nodiscard]] T &operator[](uint index) noexcept { return _m[index]; }\n"
           "    [[nodiscard]] T operator[](uint index) const noexcept { return _m[index]; }\n"
           "};\n"
           "\n"
           "template<typename T, typename U>\n"
           "T as_type(U u) noexcept { return *reinterpret_cast<T *>(&u); }\n\n";
    
    CppCodegen::emit(f);
}

void CudaCodegen::_emit_function_body(const Function &f) {
    _os << "{\n";
    for (auto &&v : f.threadgroup_variables()) {
        _os << "    __shared__ array<";
        _emit_type(v->type());
        _os << ", " << v->threadgroup_element_count() << "> v" << v->uid() << ";\n";
    }
    for (auto &&v : f.builtins()) {
        if (v->is_thread_id()) {
            _os << "    auto tid = static_cast<uint>(blockIdx.x * blockDim.x + threadIdx.x);\n";
        } else if (v->is_thread_xy()) {
            _os << "    auto txy = luisa::make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);\n";
        }
    }
    _indent = 1;
    for (auto &&stmt : f.body()->statements()) {
        _after_else = false;
        stmt->accept(*this);
    }
    _os << "}\n";
}

void CudaCodegen::_emit_function_decl(const Function &f) {
    
    auto &&args = f.arguments();
    
    if (!args.empty()) {
        _os << "struct Argument {\n";
        for (auto &&arg : args) {
            if (arg->is_immutable_argument() || arg->is_uniform_argument()) {
                _os << "    ";
                _os << "const ";
                _emit_type(arg->type());
                _os << " v" << arg->uid() << ";\n";
            }
        }
        for (auto &&arg : args) {
            if (arg->is_texture_argument()) {
                _os << "    ";
                switch (arg->texture()->format()) {
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
                _os << "v" << arg->uid() << ";\n";
            } else if (arg->is_buffer_argument()) {
                _os << "    ";
                _emit_type(arg->type());
                _os << " * __restrict__ v" << arg->uid() << ";\n";
            }
        }
        _os << "};\n\n"
            << "extern \"C\" __global__ void " << f.name() << "(const Argument arg) ";
    } else {
        _os << "extern \"C\" __global__ void " << f.name() << "() ";
    }
}

void CudaCodegen::_emit_builtin_function_name(const std::string &name) {
    if (name == "threadgroup_barrier") {
        _os << "__syncthreads";
    } else {
        _os << "luisa::";
        CppCodegen::_emit_builtin_function_name(name);
    }
}

void CudaCodegen::_emit_variable(const Variable *v) {
    if (v->is_argument()) {
        _os << "arg.v" << v->uid();
    } else {
        CppCodegen::_emit_variable(v);
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
