//
// Created by Mike Smith on 2020/7/31.
//

#include "metal_codegen.h"

namespace luisa::metal {

using namespace luisa::dsl;

void MetalCodegen::_emit_function_decl(const Function &f) {
    
    // kernel head
    _os << "kernel void " << f.name() << "(";
    auto &&args = f.arguments();
    for (auto i = 0ul; i < args.size(); i++) {
        auto &&arg = args[i];
        _emit_argument_decl(arg);
        if (i != args.size() - 1u) { _os << ", "; }
    }
    for (auto &&v : f.builtin_variables()) {
        _os << ", ";
        switch (v.builtin_tag()) {
            case BuiltinVariable::THREAD_ID:
                _os << "uint2 $tid [[thread_position_in_grid]]";
                break;
            default:
                _os << "int $unknown";
                break;
        }
    }
    _os << ") ";
}

void MetalCodegen::_emit_builtin_variable(BuiltinVariable tag) {
    switch (tag) {
        case BuiltinVariable::THREAD_ID:
            _os << "($tid.x)";
            break;
        default:
            _os << "$unknown";
            break;
    }
}

void MetalCodegen::_emit_argument_decl(Variable v) {
    auto vt = v.type();
    auto is_ptr_or_ref = (vt != nullptr) &&
                         (vt->type == TypeCatalog::POINTER || vt->type == TypeCatalog::REFERENCE ||
                          (vt->type == TypeCatalog::CONST &&
                           (vt->element_type->type == TypeCatalog::POINTER || vt->element_type->type == TypeCatalog::REFERENCE)));
    if (is_ptr_or_ref) {
        _os << "device ";
        _emit_variable_decl(v);
    } else {
        _os << "constant LUISA_UNIFORM<";
        _emit_type(vt);
        _os << "> v" << v.uid();
    }
}

void MetalCodegen::emit(const Function &f) {
    // stabs
    _os << "#include <metal_stdlib>\n"
           "\n"
           "using namespace metal;\n"
           "\n"
           "#ifndef LUISA_COMPUTE_METAL_STABS\n"
           "#define LUISA_COMPUTE_METAL_STABS\n"
           "template<typename T> using LUISA_UNIFORM = T &;\n"
           "#endif\n"
           "\n";
    CppCodegen::emit(f);
}

void MetalCodegen::_emit_type(const TypeDesc *desc) {
    if (desc != nullptr && desc->type == TypeCatalog::ARRAY) {
        _os << "array<";
        _emit_type(desc->element_type);
        _os << ", " << desc->element_count << ">";
    } else {
        CppCodegen::_emit_type(desc);
    }
}

}
