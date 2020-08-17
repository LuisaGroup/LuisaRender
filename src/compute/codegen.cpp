//
// Created by Mike Smith on 2020/7/31.
//

#include "codegen.h"

namespace luisa::compute::dsl {

void CppCodegen::emit(const Function &f) {
    
    _indent = 0;
    
    // used structs
    auto used_structs = toposort_structs(f.used_types());
    for (auto s : used_structs) { _emit_struct_fwd_decl(s); }
    
    _os << "\n";
    for (auto s : used_structs) {
        _emit_struct_decl(s);
    }
    
    _emit_argument_struct_decl(f);
    
    // function head
    _emit_function_decl(f);
    
    // function body
    _os << "{\n";
    _indent++;
    for (auto &&stmt : f.statements()) { stmt->accept(*this); }
    _os << "}\n\n";
}

void CppCodegen::visit(const UnaryExpr &unary_expr) {
    switch (unary_expr.op()) {
        case UnaryOp::PLUS:
            _os << "+";
            _emit_variable(unary_expr.operand());
            break;
        case UnaryOp::MINUS:
            _os << "-";
            _emit_variable(unary_expr.operand());
            break;
        case UnaryOp::NOT:
            _os << "!";
            _emit_variable(unary_expr.operand());
            break;
        case UnaryOp::BIT_NOT:
            _os << "~";
            _emit_variable(unary_expr.operand());
            break;
        case UnaryOp::ADDRESS_OF:
            _os << "(&";
            _emit_variable(unary_expr.operand());
            _os << ")";
            break;
        case UnaryOp::DEREFERENCE:
            _os << "(*";
            _emit_variable(unary_expr.operand());
            _os << ")";
            break;
        default:
            break;
    }
}

void CppCodegen::visit(const BinaryExpr &binary_expr) {
    
    auto op = binary_expr.op();
    if (op != BinaryOp::ACCESS) { _os << "("; }
    _emit_variable(binary_expr.lhs());
    switch (op) {
        case BinaryOp::ADD:
            _os << " + ";
            break;
        case BinaryOp::SUB:
            _os << " - ";
            break;
        case BinaryOp::MUL:
            _os << " * ";
            break;
        case BinaryOp::DIV:
            _os << " / ";
            break;
        case BinaryOp::MOD:
            _os << " % ";
            break;
        case BinaryOp::BIT_AND:
            _os << " & ";
            break;
        case BinaryOp::BIT_OR:
            _os << " | ";
            break;
        case BinaryOp::BIT_XOR:
            _os << " ^ ";
            break;
        case BinaryOp::SHL:
            _os << " << ";
            break;
        case BinaryOp::SHR:
            _os << " >> ";
            break;
        case BinaryOp::AND:
            _os << " && ";
            break;
        case BinaryOp::OR:
            _os << " || ";
            break;
        case BinaryOp::LESS:
            _os << " < ";
            break;
        case BinaryOp::GREATER:
            _os << " > ";
            break;
        case BinaryOp::LESS_EQUAL:
            _os << " <= ";
            break;
        case BinaryOp::GREATER_EQUAL:
            _os << " >= ";
            break;
        case BinaryOp::EQUAL:
            _os << " == ";
            break;
        case BinaryOp::NOT_EQUAL:
            _os << " != ";
            break;
        case BinaryOp::ACCESS:
            _os << "[";
            break;
        case BinaryOp::ASSIGN:
            _os << " = ";
            break;
        case BinaryOp::ADD_ASSIGN:
            _os << " += ";
            break;
        case BinaryOp::SUB_ASSIGN:
            _os << " -= ";
            break;
        case BinaryOp::MUL_ASSIGN:
            _os << " *= ";
            break;
        case BinaryOp::DIV_ASSIGN:
            _os << " /= ";
            break;
        case BinaryOp::MOD_ASSIGN:
            _os << " %= ";
            break;
        case BinaryOp::BIT_AND_ASSIGN:
            _os << " &= ";
            break;
        case BinaryOp::BIT_OR_ASSIGN:
            _os << " |= ";
            break;
        case BinaryOp::BIT_XOR_ASSIGN:
            _os << " ^= ";
            break;
        case BinaryOp::SHL_ASSIGN:
            _os << " <<= ";
            break;
        case BinaryOp::SHR_ASSIGN:
            _os << " >>= ";
            break;
        default:
            break;
    }
    _emit_variable(binary_expr.rhs());
    if (op == BinaryOp::ACCESS) { _os << "]"; } else { _os << ")"; }
}

void CppCodegen::visit(const MemberExpr &member_expr) {
    _os << "(";
    _emit_variable(member_expr.self());
    _os << "." << member_expr.member() << ")";
}

void CppCodegen::visit(const LiteralExpr &literal_expr) {
    
    auto &&values = literal_expr.values();
    auto flags = _os.flags();
    _os << std::boolalpha << std::hexfloat;
    for (auto i = 0ul; i < values.size(); i++) {
        std::visit([this](auto &&v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Variable>) { _emit_variable(v); }
            else if constexpr (std::is_same_v<T, bool>) { _os << v; }
            else if constexpr (std::is_same_v<T, float>) { _os << v << "f"; }
            else if constexpr (std::is_same_v<T, int8_t>) { _os << "static_cast<int8_t>(" << v << ")"; }
            else if constexpr (std::is_same_v<T, uint8_t>) { _os << "static_cast<uint8_t>(" << v << ")"; }
            else if constexpr (std::is_same_v<T, int16_t>) { _os << "static_cast<int16_t>(" << v << ")"; }
            else if constexpr (std::is_same_v<T, uint16_t>) { _os << "static_cast<uint16_t>(" << v << ")"; }
            else if constexpr (std::is_same_v<T, int32_t>) { _os << v; }
            else if constexpr (std::is_same_v<T, uint32_t>) { _os << v << "u"; }
            else if constexpr (std::is_same_v<T, int64_t>) { _os << v << "ll"; }
            else if constexpr (std::is_same_v<T, uint64_t>) { _os << v << "ull"; }
        }, values[i]);
        if (i != values.size() - 1u) { _os << ", "; }
    }
    _os.flags(flags);
}

void CppCodegen::visit(const CallExpr &call_expr) {
    _emit_function_call(call_expr.name());
    _os << "(";
    auto &&args = call_expr.arguments();
    for (auto i = 0ul; i < args.size(); i++) {
        _emit_variable(args[i]);
        if (i != args.size() - 1u) { _os << ", "; }
    }
    if (call_expr.name().find("atomic_") == 0) {
        _os << ", memory_order_relaxed";
        if (call_expr.name() == "atomic_compare_exchange_weak_explicit") {
            _os << ", memory_order_relaxed";
        }
    }
    _os << ")";
}

void CppCodegen::visit(const CastExpr &cast_expr) {
    switch (cast_expr.op()) {
        case CastOp::STATIC:
            _os << "static_cast";
            break;
        case CastOp::REINTERPRET:
            _os << "reinterpret_cast";
            break;
        case CastOp::BITWISE:
            _os << "as";
            break;
        default:
            break;
    }
    _os << "<";
    _emit_type(cast_expr.dest_type());
    _os << ">(";
    _emit_variable(cast_expr.source());
    _os << ")";
}

void CppCodegen::visit(const DeclareStmt &declare_stmt) {
    _emit_indent();
    if (declare_stmt.is_constant()) { _os << "constexpr "; }
    _emit_variable_decl(declare_stmt.var());
    _os << "{";
    _emit_variable(declare_stmt.initialization());
    _os << "};\n";
}

void CppCodegen::visit(const KeywordStmt &stmt) {
    auto &&kw = stmt.keyword();
    if (kw == "}") { _indent--; }
    if (kw != "{") { _emit_indent(); }
    _os << kw;
    if (kw == "else" || kw == "do") { _os << " "; } else { _os << "\n"; }
    if (kw == "{") { _indent++; }
}

void CppCodegen::visit(const IfStmt &if_stmt) {
    if (!if_stmt.is_elif()) { _emit_indent(); }
    _os << "if (";
    _emit_variable(if_stmt.condition());
    _os << ") ";
}

void CppCodegen::visit(const SwitchStmt &switch_stmt) {
    _emit_indent();
    _os << "switch (";
    _emit_variable(switch_stmt.expression());
    _os << ") ";
}

void CppCodegen::visit(const CaseStmt &case_stmt) {
    _emit_indent();
    if (case_stmt.is_default()) {
        _os << "default: ";
    } else {
        _os << "case ";
        _emit_variable(case_stmt.expression());
        _os << ": ";
    }
}

void CppCodegen::visit(const WhileStmt &while_stmt) {
    _emit_indent();
    _os << "while (";
    _emit_variable(while_stmt.condition());
    _os << ")";
    if (while_stmt.is_do_while()) { _os << ";\n"; } else { _os << " "; }
}

void CppCodegen::visit(const ForStmt &loop_stmt) {
    _emit_indent();
    _os << "for (; ";
    _emit_variable(loop_stmt.i());
    _os << " < ";
    _emit_variable(loop_stmt.end());
    _os << "; ";
    _emit_variable(loop_stmt.i());
    _os << " += ";
    _emit_variable(loop_stmt.step());
    _os << ") ";
}

void CppCodegen::visit(const ExprStmt &expr_stmt) {
    _emit_indent();
    _emit_variable(expr_stmt.expression());
    _os << ";\n";
}

void CppCodegen::_emit_struct_decl(const TypeDesc *desc) {
    
    _os << "struct alignas(" << desc->alignment << ") Struct$" << desc->uid << " {";
    if (!desc->member_names.empty()) { _os << "\n"; }
    
    // for each member
    for (auto i = 0u; i < desc->member_names.size(); i++) {
        _os << "    ";
        _emit_type(desc->member_types[i]);
        if (auto mt = desc->member_types[i];
            mt != nullptr &&
            mt->type != TypeCatalog::REFERENCE && mt->type != TypeCatalog::POINTER) { _os << " "; }
        _os << desc->member_names[i] << ";\n";
    }
    _os << "};\n\n";
}

void CppCodegen::_emit_variable_decl(Variable v) {
    _emit_type(v.type());
    if (v.type() != nullptr &&
        v.type()->type != TypeCatalog::POINTER &&
        v.type()->type != TypeCatalog::REFERENCE) { _os << " "; }
    _os << "v" << v.uid();
}

void CppCodegen::_emit_variable(Variable v) {
    if (v.is_temporary()) {
        v.expression()->accept(*this);
    } else if (v.is_argument()) {
        _os << "arg.v" << v.uid();
    } else if (v.is_local()) {
        _os << "v" << v.uid();
    } else if (v.is_thread_id()) {
        _os << "$tid";
    } else if (v.is_thread_xy()) {
        _os << "$txy";
    } else if (v.is_thread_xyz()) {
        _os << "$txyz";
    } else {
        _os << "$unknown";
    }
}

void CppCodegen::_emit_indent() {
    for (auto i = 0; i < _indent; i++) { _os << "    "; }
}

void CppCodegen::_emit_type(const TypeDesc *desc) {
    
    if (desc == nullptr) {
        _os << "[MISSING]";
        return;
    }
    
    switch (desc->type) {
        case TypeCatalog::UNKNOWN:
            _os << "[UNKNOWN]";
            break;
        case TypeCatalog::AUTO:
            _os << "auto";
            break;
        case TypeCatalog::BOOL:
            _os << "bool";
            break;
        case TypeCatalog::FLOAT:
            _os << "float";
            break;
        case TypeCatalog::INT8:
            _os << "byte";
            break;
        case TypeCatalog::UINT8:
            _os << "ubyte";
            break;
        case TypeCatalog::INT16:
            _os << "short";
            break;
        case TypeCatalog::UINT16:
            _os << "ushort";
            break;
        case TypeCatalog::INT32:
            _os << "int";
            break;
        case TypeCatalog::UINT32:
            _os << "uint";
            break;
        case TypeCatalog::INT64:
            _os << "long";
            break;
        case TypeCatalog::UINT64:
            _os << "ulong";
            break;
        case TypeCatalog::VECTOR2:
            _emit_type(desc->element_type);
            _os << 2;
            break;
        case TypeCatalog::VECTOR3:
            _emit_type(desc->element_type);
            _os << 3;
            break;
        case TypeCatalog::VECTOR4:
            _emit_type(desc->element_type);
            _os << 4;
            break;
        case TypeCatalog::VECTOR3_PACKED:
            _os << "packed_";
            _emit_type(desc->element_type);
            _os << 2;
            break;
        case TypeCatalog::MATRIX3:
            _os << "float3x3";
            break;
        case TypeCatalog::MATRIX4:
            _os << "float4x4";
            break;
        case TypeCatalog::ARRAY:
            _os << "array<";
            _emit_type(desc->element_type);
            _os << ", " << desc->element_count << ">";
            break;
        case TypeCatalog::CONST:
            _emit_type(desc->element_type);
            if (auto et = desc->element_type;
                et != nullptr &&
                et->type != TypeCatalog::POINTER &&
                et->type != TypeCatalog::REFERENCE) { _os << " "; }
            _os << "const";
            break;
        case TypeCatalog::POINTER:
            _emit_type(desc->element_type);
            if (auto et = desc->element_type;
                et != nullptr &&
                et->type != TypeCatalog::POINTER &&
                et->type != TypeCatalog::REFERENCE) { _os << " "; }
            _os << "*";
            break;
        case TypeCatalog::REFERENCE:
            _emit_type(desc->element_type);
            if (auto et = desc->element_type;
                et != nullptr &&
                et->type != TypeCatalog::POINTER &&
                et->type != TypeCatalog::REFERENCE) { _os << " "; }
            _os << "&";
            break;
        case TypeCatalog::TEXTURE:
            _os << "texture2d<float, access::";
            if (desc->access == TextureAccess::READ) { _os << "read"; }
            else if (desc->access == TextureAccess::WRITE) { _os << "write"; }
            else if (desc->access == TextureAccess::READ_WRITE) { _os << "read_write"; }
            else if (desc->access == TextureAccess::SAMPLE) { _os << "sample"; }
            _os << ">";
            break;
        case TypeCatalog::ATOMIC:
            _os << "atomic<";
            _emit_type(desc->element_type);
            _os << ">";
        case TypeCatalog::STRUCTURE:
            _os << "Struct$" << desc->uid;
            break;
        default:
            _os << "[BAD]";
            break;
    }
}

void CppCodegen::_emit_function_decl(const Function &f) {
    _os << "void " << f.name() << "(const Argument &arg) ";
}

void CppCodegen::_emit_function_call(const std::string &name) {
    _os << name;
}

void CppCodegen::_emit_struct_fwd_decl(const TypeDesc *desc) {
    _os << "struct Struct$" << desc->uid << ";\n";
}

void CppCodegen::_emit_argument_struct_decl(const Function &f) {
    _os << "struct Argument {\n";
    for (auto &&arg : f.arguments()) {
        _os << "    ";
        _emit_argument_member_decl(arg);
        _os << ";\n";
    }
    _os << "};\n\n";
}

void CppCodegen::_emit_argument_member_decl(Variable v) {
    _emit_variable_decl(std::move(v));
}

}
