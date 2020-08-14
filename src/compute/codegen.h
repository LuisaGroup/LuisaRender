//
// Created by Mike Smith on 2020/7/31.
//

#pragma once

#include <ostream>
#include <compute/function.h>

namespace luisa {
class Device;
}

namespace luisa::compute::dsl {

class Codegen : Noncopyable {

protected:
    std::ostream &_os;

public:
    explicit Codegen(std::ostream &os) noexcept : _os{os} {}
    virtual ~Codegen() noexcept = default;
    virtual void emit(const Function &function) = 0;
};

// Example codegen for C++
class CppCodegen : public Codegen, public ExprVisitor, public StmtVisitor {

protected:
    int32_t _indent{0};
    
    virtual void _emit_indent();
    virtual void _emit_function_decl(const Function &f);
    virtual void _emit_struct_decl(const TypeDesc *desc);
    virtual void _emit_struct_fwd_decl(const TypeDesc *desc);
    virtual void _emit_variable(Variable v);
    virtual void _emit_variable_decl(Variable v);
    virtual void _emit_argument_member_decl(Variable v);
    virtual void _emit_type(const TypeDesc *desc);
    virtual void _emit_function_call(const std::string &name);
    virtual void _emit_argument_struct_decl(const compute::dsl::Function &f);

public:
    explicit CppCodegen(std::ostream &os) noexcept : Codegen{os} {}
    void emit(const Function &f) override;
    
    // expression visitors
    void visit(const UnaryExpr &unary_expr) override;
    void visit(const BinaryExpr &binary_expr) override;
    void visit(const MemberExpr &member_expr) override;
    void visit(const LiteralExpr &v) override;
    void visit(const CallExpr &call_expr) override;
    void visit(const CastExpr &cast_expr) override;
    
    // statement visitors
    void visit(const DeclareStmt &declare_stmt) override;
    void visit(const KeywordStmt &stmt) override;
    void visit(const IfStmt &if_stmt) override;
    void visit(const WhileStmt &while_stmt) override;
    void visit(const ForStmt &loop_stmt) override;
    void visit(const ExprStmt &expr_stmt) override;
    void visit(const SwitchStmt &switch_stmt) override;
    void visit(const CaseStmt &case_stmt) override;
};

}
