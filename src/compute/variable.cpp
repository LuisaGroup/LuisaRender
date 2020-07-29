//
// Created by Mike Smith on 2020/7/10.
//

#include <compute/expression.h>
#include <compute/statement.h>
#include <compute/function.h>

#include "variable.h"

namespace luisa::dsl {

Variable::Variable(Function *func, const TypeDesc *type, uint32_t uid) noexcept
    : _function{func}, _type{type}, _uid{uid} {}

Variable::Variable(Expression *expr) noexcept
    : _function{expr->function()}, _expression{expr} {}

#define MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(op, op_tag)                                         \
Variable Variable::operator op(Variable rhs) const noexcept {                                      \
    return _function->add_expression(std::make_unique<BinaryExpr>(BinaryOp::op_tag, *this, rhs));  \
}                                                                                                  \

MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(+, ADD)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(-, SUB)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(*, MUL)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(/, DIV)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(%, MOD)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(<<, SHL)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(>>, SHR)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(&, BIT_AND)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(|, BIT_OR)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(^, BIT_XOR)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(&&, AND)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(||, OR)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(==, EQUAL)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(!=, NOT_EQUAL)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(<, LESS)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(>, GREATER)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(<=, LESS_EQUAL)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(>=, GREATER_EQUAL)
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD([], ACCESS)

#undef MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD

Variable::Variable(Function *func, const TypeDesc *type, BuiltinTag tag) noexcept
    : _function{func}, _type{type}, _builtin_tag{tag} {}

#define MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(op, op_tag)  \
void Variable::operator op(Variable rhs) const noexcept {  \
    _function->add_statement(std::make_unique<AssignStmt>(AssignOp::op_tag, *this, rhs._expression));  \
}

MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(=, ASSIGN)
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(+=, ADD_ASSIGN)
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(-=, SUB_ASSIGN)
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(*=, MUL_ASSIGN)
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(/=, DIV_ASSIGN)
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(%=, MOD_ASSIGN)
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(&=, BIT_AND_ASSIGN)
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(|=, BIT_OR_ASSIGN)
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(^=, BIT_XOR_ASSIGN)
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(<<=, SHL_ASSIGN)
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(>>=, SHR_ASSIGN)

#undef MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD

Variable Variable::member(std::string m) const noexcept {
    return _function->add_expression(std::make_unique<MemberExpr>(*this, std::move(m)));
}

Variable Variable::arrow(std::string m) const noexcept {
    return _function->add_expression(std::make_unique<ArrowExpr>(*this, std::move(m)));
}
    
}
