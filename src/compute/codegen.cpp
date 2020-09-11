//
// Created by Mike Smith on 2020/7/31.
//

#include "codegen.h"

namespace luisa::compute::dsl {

void CppCodegen::emit(const Function &f) {
    
    // used structs
    std::vector<const TypeDesc *> used_structures{f.used_structures().cbegin(), f.used_structures().cend()};
    std::sort(used_structures.begin(), used_structures.end(), [](const TypeDesc *lhs, const TypeDesc *rhs) noexcept {
        return lhs->uid() < rhs->uid();
    });
    
    for (auto s : used_structures) {
        _emit_struct_decl(s);
    }
    
    // function head
    _emit_function_decl(f);
    _emit_function_body(f);
}

void CppCodegen::visit(const ValueExpr *literal_expr) {
    auto &&value = literal_expr->value();
    auto flags = _os.flags();
    _os << std::boolalpha << std::hexfloat;
    std::visit([this](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, Variable>) { _emit_variable(v); }
        else if constexpr (std::is_same_v<T, bool>) { _os << v; }
        else if constexpr (std::is_same_v<T, float>) {
            if (std::isinf(v)) { _os << "static_cast<float>(1.0f / +0.0f)"; } else { _os << v << "f"; }  // TODO: Better handling of inf/nan
        } else if constexpr (std::is_same_v<T, int8_t>) { _os << "static_cast<int8_t>(" << v << ")"; }
        else if constexpr (std::is_same_v<T, uint8_t>) { _os << "static_cast<uint8_t>(" << v << ")"; }
        else if constexpr (std::is_same_v<T, int16_t>) { _os << "static_cast<int16_t>(" << v << ")"; }
        else if constexpr (std::is_same_v<T, uint16_t>) { _os << "static_cast<uint16_t>(" << v << ")"; }
        else if constexpr (std::is_same_v<T, int32_t>) { _os << v; }
        else if constexpr (std::is_same_v<T, uint32_t>) { _os << v << "u"; }
    }, value);
    _os.flags(flags);
}

void CppCodegen::_emit_struct_decl(const TypeDesc *desc) {
    
    _os << "struct alignas(" << desc->alignment << ") Struct_" << desc->uid() << " {";
    if (!desc->member_names.empty()) { _os << "\n"; }
    
    // for each member
    for (auto i = 0u; i < desc->member_names.size(); i++) {
        _os << "    ";
        _emit_type(desc->member_types[i]);
        _os << " " << desc->member_names[i] << ";\n";
    }
    _os << "};\n\n";
}

void CppCodegen::_emit_variable(const Variable *v) {
    if (v->is_temporary()) {
        v->expression()->accept(*this);
    } else if (v->is_argument() || v->is_local() || v->is_threadgroup()) {
        _os << "v" << v->uid();
    } else if (v->is_thread_id()) {
        _os << "tid";
    } else if (v->is_thread_xy()) {
        _os << "txy";
    } else {
        _os << "unknown";
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
        case TypeCatalog::BOOL:
            _os << "bool";
            break;
        case TypeCatalog::FLOAT:
            _os << "float";
            break;
        case TypeCatalog::INT8:
            _os << "char";
            break;
        case TypeCatalog::UINT8:
            _os << "uchar";
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
        case TypeCatalog::ATOMIC:
            _os << "atomic<";
            _emit_type(desc->element_type);
            _os << ">";
        case TypeCatalog::STRUCTURE:
            _os << "Struct_" << desc->uid();
            break;
        default:
            _os << "[BAD]";
            break;
    }
}

void CppCodegen::_emit_function_decl(const Function &f) {
    _os << "void " << f.name() << "(";
    auto &&args = f.arguments();
    for (auto i = 0u; i < args.size(); i++) {
        auto arg = args[i].get();
        if (arg->is_texture_argument()) {
            auto usage = f.texture_usage(arg->texture());
            auto read = static_cast<bool>(usage & Function::texture_read_bit);
            auto write = static_cast<bool>(usage & Function::texture_write_bit);
            auto sample = static_cast<bool>(usage & Function::texture_sample_bit);
            assert(!(sample && (read || write)));
            if (read && write) {
                _os << "texture2d<float, access::read_write> v" << arg->uid();
            } else if (read) {
                _os << "texture2d<float, access::read> v" << arg->uid();
            } else if (write) {
                _os << "texture2d<float, access::write> v" << arg->uid();
            } else if (sample) {
                _os << "texture2d<float, access::sample> v" << arg->uid();
            } else { continue; }
            if (i != args.size() - 1u) { _os << ", "; }
        } else if (arg->is_buffer_argument()) {
            _emit_type(arg->type());
            _os << " *v" << arg->uid();
            if (i != args.size() - 1u) { _os << ", "; }
        } else if (arg->is_immutable_argument() || arg->is_uniform_argument()) {
            _emit_type(arg->type());
            _os << " &v" << arg->uid();
            if (i != args.size() - 1u) { _os << ", "; }
        }
    }
    _os << ") ";
}

void CppCodegen::_emit_function_body(const Function &f) {
    _os << "{\n";
    for (auto &&v : f.threadgroup_variables()) {
        _os << "    threadgroup array<";
        _emit_type(v->type());
        _os << ", " << v->threadgroup_element_count() << "> v" << v->uid() << ";\n";
    }
    _indent = 1;
    for (auto &&stmt : f.body()->statements()) {
        _after_else = false;
        stmt->accept(*this);
    }
    _os << "}\n";
}

void CppCodegen::visit(const UnaryExpr *unary_expr) {
    switch (unary_expr->op()) {
        case UnaryOp::PLUS:
            _os << "+";
            _emit_variable(unary_expr->operand());
            break;
        case UnaryOp::MINUS:
            _os << "-";
            _emit_variable(unary_expr->operand());
            break;
        case UnaryOp::NOT:
            _os << "!";
            _emit_variable(unary_expr->operand());
            break;
        case UnaryOp::BIT_NOT:
            _os << "~";
            _emit_variable(unary_expr->operand());
            break;
        default:
            break;
    }
}

void CppCodegen::visit(const BinaryExpr *binary_expr) {
    auto op = binary_expr->op();
    if (op != BinaryOp::ACCESS) { _os << "("; }
    _emit_variable(binary_expr->lhs());
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
        default:
            break;
    }
    _emit_variable(binary_expr->rhs());
    if (op == BinaryOp::ACCESS) { _os << "]"; } else { _os << ")"; }
}

void CppCodegen::visit(const MemberExpr *member_expr) {
    _os << "(";
    _emit_variable(member_expr->self());
    _os << "." << member_expr->member() << ")";
}

void CppCodegen::visit(const CallExpr *func_expr) {
    _emit_builtin_function_name(func_expr->name());
    _os << "(";
    auto &&args = func_expr->arguments();
    for (auto i = 0u; i < args.size(); i++) {
        auto arg = args[i];
        _emit_variable(arg);
        if (i != args.size() - 1u) { _os << ", "; }
    }
    _os << ")";
}

void CppCodegen::visit(const CastExpr *cast_expr) {
    switch (cast_expr->op()) {
        case CastOp::STATIC:
            _os << "static_cast<";
            _emit_type(cast_expr->dest_type());
            _os << ">(";
            _emit_variable(cast_expr->source());
            _os << ")";
            break;
        case CastOp::REINTERPRET:
            _os << "reinterpret_cast<device ";
            _emit_type(cast_expr->dest_type());
            _os << " &>(";
            _emit_variable(cast_expr->source());
            _os << ")";
            break;
        case CastOp::BITWISE:
            _os << "as<";
            _emit_type(cast_expr->dest_type());
            _os << ">(";
            _emit_variable(cast_expr->source());
            _os << ")";
            break;
    }
}

void CppCodegen::visit(const TextureExpr *tex_expr) {
    switch (tex_expr->op()) {
        case TextureOp::READ:
            _os << "read(";
            _emit_variable(tex_expr->texture());
            _os << ", ";
            _emit_variable(tex_expr->coord());
            _os << ")";
            break;
        case TextureOp::WRITE:
            _os << "write(";
            _emit_variable(tex_expr->texture());
            _os << ", ";
            _emit_variable(tex_expr->coord());
            _os << ", ";
            _emit_variable(tex_expr->value());
            _os << ")";
            break;
        case TextureOp::SAMPLE:
            LUISA_ERROR("Not implemented!");
            break;
    }
}

void CppCodegen::visit(const EmptyStmt *stmt) {
    _emit_indent();
    _os << ";\n";
}

void CppCodegen::visit(const BreakStmt *stmt) {
    _emit_indent();
    _os << "break;\n";
}

void CppCodegen::visit(const ContinueStmt *stmt) {
    _emit_indent();
    _os << "continue;\n";
}

void CppCodegen::visit(const ReturnStmt *stmt) {
    _emit_indent();
    _os << "return;\n";
}

void CppCodegen::visit(const ScopeStmt *scope_stmt) {
    if (scope_stmt->statements().empty()) {
        _os << "{}";
    } else {
        _os << "{\n";
        _indent++;
        for (auto &&stmt : scope_stmt->statements()) {
            _after_else = false;
            stmt->accept(*this);
        }
        _indent--;
        _emit_indent();
        _os << "}";
    }
}

void CppCodegen::visit(const DeclareStmt *declare_stmt) {
    _emit_indent();
    auto v = declare_stmt->var();
    _emit_type(v->type());
    _os << " v" << v->uid() << "{";
    auto &&inits = declare_stmt->init_expr();
    for (auto i = 0u; i < inits.size(); i++) {
        _emit_variable(inits[i]);
        if (i != inits.size() - 1u) { _os << ", "; }
    }
    _os << "};\n";
}

void CppCodegen::visit(const IfStmt *if_stmt) {
    if (!_after_else) { _emit_indent(); }
    _os << "if (";
    _emit_variable(if_stmt->condition());
    _os << ") ";
    visit(if_stmt->true_branch());
    if (auto fb = if_stmt->false_branch(); fb != nullptr && !fb->statements().empty()) {
        _os << " else ";
        auto &&stmts = fb->statements();
        if (stmts.size() == 1u && dynamic_cast<const IfStmt *>(stmts.front().get())) {
            _after_else = true;
            stmts.front()->accept(*this);
        } else {
            visit(if_stmt->false_branch());
            _os << "\n";
        }
    } else {
        _os << "\n";
    }
}

void CppCodegen::visit(const WhileStmt *while_stmt) {
    _emit_indent();
    _os << "while (";
    _emit_variable(while_stmt->condition());
    _os << ") ";
    visit(while_stmt->body());
    _os << "\n";
}

void CppCodegen::visit(const DoWhileStmt *do_while_stmt) {
    _emit_indent();
    _os << "do ";
    visit(do_while_stmt->body());
    _os << " while (";
    _emit_variable(do_while_stmt->condition());
    _os << ");\n";
}

void CppCodegen::visit(const ExprStmt *expr_stmt) {
    _emit_indent();
    expr_stmt->expr()->accept(*this);
    _os << ";\n";
}

void CppCodegen::visit(const SwitchStmt *switch_stmt) {
    _emit_indent();
    _os << "switch (";
    _emit_variable(switch_stmt->expr());
    _os << ") ";
    visit(switch_stmt->body());
    _os << "\n";
}

void CppCodegen::visit(const SwitchCaseStmt *case_stmt) {
    _emit_indent();
    _os << "case ";
    _emit_variable(case_stmt->expr());
    _os << ": ";
    visit(case_stmt->body());
    _os << "\n";
}

void CppCodegen::visit(const SwitchDefaultStmt *default_stmt) {
    _emit_indent();
    _os << "default: ";
    visit(default_stmt->body());
    _os << "\n";
}

void CppCodegen::visit(const AssignStmt *assign_stmt) {
    _emit_indent();
    _emit_variable(assign_stmt->lhs());
    switch (assign_stmt->op()) {
        case AssignOp::ASSIGN:
            _os << " = ";
            break;
        case AssignOp::ADD_ASSIGN:
            _os << " += ";
            break;
        case AssignOp::SUB_ASSIGN:
            _os << " -= ";
            break;
        case AssignOp::MUL_ASSIGN:
            _os << " *= ";
            break;
        case AssignOp::DIV_ASSIGN:
            _os << " /= ";
            break;
        case AssignOp::MOD_ASSIGN:
            _os << " %= ";
            break;
        case AssignOp::BIT_AND_ASSIGN:
            _os << " &= ";
            break;
        case AssignOp::BIT_OR_ASSIGN:
            _os << " |= ";
            break;
        case AssignOp::BIT_XOR_ASSIGN:
            _os << " ^= ";
            break;
        case AssignOp::SHL_ASSIGN:
            _os << " <<= ";
            break;
        case AssignOp::SHR_ASSIGN:
            _os << " >>= ";
            break;
    }
    _emit_variable(assign_stmt->rhs());
    _os << ";\n";
}

void CppCodegen::_emit_builtin_function_name(const std::string &func) {
    _os << func;
}

}
