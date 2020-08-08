//
// Created by Mike Smith on 2020/7/31.
//

#include "metal_codegen.h"

namespace luisa::metal {

using namespace luisa::compute::dsl;

void MetalCodegen::_emit_function_decl(const Function &f) {
    
    // kernel head
    _os << "kernel void " << f.name() << "(device const Argument &arg [[buffer(0)]]";
    for (auto &&v : f.builtin_variables()) {
        _os << ", ";
        switch (v.builtin_tag()) {
            case BuiltinVariable::THREAD_ID:
                _os << "uint $tid [[thread_position_in_grid]]";
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
            _os << "$tid";
            break;
        default:
            _os << "$unknown";
            break;
    }
}

void MetalCodegen::emit(const Function &f) {
    // stabs
    _os << "#include <metal_stdlib>\n"
           "\n"
           "using namespace metal;\n"
           "\n"
           "template<typename C, typename T, typename F> inline auto ite(C p, T t, F f) { return select(t, f, p); }\n"
           "\n";
    CppCodegen::emit(f);
}

void MetalCodegen::_emit_type(const TypeDesc *desc) {
    if (is_ptr_or_ref(desc)) { _os << "device "; }
    if (desc->type == TypeCatalog::ATOMIC) {
        _os << "_atomic<";
        _emit_type(desc->element_type);
        _os << ">";
    } else {
        CppCodegen::_emit_type(desc);
    }
}

void MetalCodegen::_emit_function_call(const std::string &name) {
    if (name.find("make_") == 0u) {
        _os << std::string_view{name.c_str()}.substr(5);
    } else if (name == "lerp") {
        _os << "mix";
    } else if (name == "select") {
        _os << "ite";
    } else {
        _os << name;
    }
}

}
