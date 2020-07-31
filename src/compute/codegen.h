//
// Created by Mike Smith on 2020/7/31.
//

#pragma once

#include <ostream>
#include <compute/function.h>

namespace luisa {
class Device;
}

namespace luisa::dsl {

class Codegen {

protected:
    std::ostream &_os;
    Device *_device{nullptr};

public:
    Codegen(std::ostream &os, Device *device) noexcept : _os{os}, _device{device} {}
    virtual ~Codegen() noexcept = default;
    virtual void emit(const Function &function) = 0;
    [[nodiscard]] Device *device() const noexcept { return _device; }
};

// Example codegen for C++
class CppCodegen : public Codegen, public ExprVisitor, public StmtVisitor {

protected:
    mutable int32_t _indent{0};
    
    virtual void _emit_indent();
    virtual void _emit_function_decl(const Function &f);
    virtual void _emit_struct_fwd_decl(const TypeDesc *desc);
    virtual void _emit_struct_decl(const TypeDesc *desc);
    virtual void _emit_variable(Variable v);
    virtual void _emit_builtin_variable(BuiltinVariable tag);
    virtual void _emit_variable_decl(Variable v);
    virtual void _emit_argument_decl(Variable v) { _emit_variable_decl(std::move(v)); }
    virtual void _emit_type(const TypeDesc *desc);
    virtual void _emit_function_call(const std::string &name);

public:
    CppCodegen(std::ostream &os, Device *device) noexcept : Codegen{os, device} {}
    void emit(const Function &f) override;
    
    // expression visitors
    void visit(const UnaryExpr &unary_expr) override;
    void visit(const BinaryExpr &binary_expr) override;
    void visit(const MemberExpr &member_expr) override;
    void visit(const ArrowExpr &arrow_expr) override;
    void visit(const LiteralExpr &v) override;
    void visit(const CallExpr &call_expr) override;
    void visit(const CastExpr &cast_expr) override;
    
    // statement visitors
    void visit(const DeclareStmt &declare_stmt) override;
    void visit(const KeywordStmt &stmt) override;
    void visit(const IfStmt &if_stmt) override;
    void visit(const WhileStmt &while_stmt) override;
    void visit(const LoopStmt &loop_stmt) override;
    void visit(const ExprStmt &expr_stmt) override;
};

}
