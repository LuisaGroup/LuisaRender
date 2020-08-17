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
        if (v.is_thread_id()) {
            _os << ", ";
            _os << "uint $tid [[thread_position_in_grid]]";
        } else if (v.is_thread_xy()) {
            _os << ", ";
            _os << "uint2 $txy [[thread_position_in_grid]]";
        } else if (v.is_thread_xyz()) {
            _os << ", ";
            _os << "uint3 $txyz [[thread_position_in_grid]]";
        }
    }
    _os << ") ";
}

void MetalCodegen::emit(const Function &f) {
    // stabs
    _os << "#include <metal_stdlib>\n"
           "\n"
           "using namespace metal;\n"
           "\n"
           "template<typename T, typename F> inline auto ite(bool p, T t, F f) { return p ? t : f; }\n"
           "\n"
           "template<access a>\n"
           "inline float4 read(texture2d<float, a> t, uint2 coord) { return t.read(coord); }\n"
           "\n"
           "template<access a>\n"
           "inline void write(texture2d<float, a> t, uint2 coord, float4 v) { return t.write(v, coord); }\n"
           "\n"
           "inline auto inverse(float3x3 m) {  // from GLM\n"
           "    auto one_over_determinant = 1.0f / (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -\n"
           "                                        m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) +\n"
           "                                        m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));\n"
           "    return float3x3(\n"
           "        (m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,\n"
           "        (m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,\n"
           "        (m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,\n"
           "        (m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,\n"
           "        (m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,\n"
           "        (m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,\n"
           "        (m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,\n"
           "        (m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,\n"
           "        (m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant);\n"
           "}\n"
           "\n"
           "inline auto inverse(float4x4 m) {  // from GLM\n"
           "    auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;\n"
           "    auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;\n"
           "    auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;\n"
           "    auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;\n"
           "    auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;\n"
           "    auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;\n"
           "    auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;\n"
           "    auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;\n"
           "    auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;\n"
           "    auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;\n"
           "    auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;\n"
           "    auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;\n"
           "    auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;\n"
           "    auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;\n"
           "    auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;\n"
           "    auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;\n"
           "    auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;\n"
           "    auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;\n"
           "    auto fac0 = float4(coef00, coef00, coef02, coef03);\n"
           "    auto fac1 = float4(coef04, coef04, coef06, coef07);\n"
           "    auto fac2 = float4(coef08, coef08, coef10, coef11);\n"
           "    auto fac3 = float4(coef12, coef12, coef14, coef15);\n"
           "    auto fac4 = float4(coef16, coef16, coef18, coef19);\n"
           "    auto fac5 = float4(coef20, coef20, coef22, coef23);\n"
           "    auto Vec0 = float4(m[1].x, m[0].x, m[0].x, m[0].x);\n"
           "    auto Vec1 = float4(m[1].y, m[0].y, m[0].y, m[0].y);\n"
           "    auto Vec2 = float4(m[1].z, m[0].z, m[0].z, m[0].z);\n"
           "    auto Vec3 = float4(m[1].w, m[0].w, m[0].w, m[0].w);\n"
           "    auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;\n"
           "    auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;\n"
           "    auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;\n"
           "    auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;\n"
           "    auto sign_a = float4(+1, -1, +1, -1);\n"
           "    auto sign_b = float4(-1, +1, -1, +1);\n"
           "    auto inv = float4x4(inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b);\n"
           "    auto dot0 = m[0] * float4(inv[0].x, inv[1].x, inv[2].x, inv[3].x);\n"
           "    auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;\n"
           "    auto one_over_determinant = 1.0f / dot1;\n"
           "    return inv * one_over_determinant;\n"
           "}\n"
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
