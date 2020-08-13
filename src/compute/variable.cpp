//
// Created by Mike Smith on 2020/7/10.
//

#include <compute/expression.h>
#include <compute/statement.h>
#include <compute/function.h>
#include <compute/stmt_helpers.h>

#include "variable.h"

namespace luisa::compute::dsl {

#define MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD_IMPL(op, op_tag)                                             \
Variable Variable::operator op(Variable rhs) const noexcept {                                               \
    return Function::current().add_expression(std::make_unique<BinaryExpr>(BinaryOp::op_tag, *this, rhs));  \
}                                                                                                           \

#define MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD(op_and_tag) \
MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD_IMPL op_and_tag

LUISA_MAP_MACRO(
    MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD,
    (+, ADD), (-, SUB), (*, MUL), (/, DIV), (%, MOD),
    (<<, SHL), (>>, SHR), (&, BIT_AND), (|, BIT_OR), (^, BIT_XOR),
    (&&, AND), (||, OR),
    (==, EQUAL), (!=, NOT_EQUAL), (<, LESS), (>, GREATER), (<=, LESS_EQUAL), (>=, GREATER_EQUAL),
    ([], ACCESS))

#undef MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD
#undef MAKE_VARIABLE_BINARY_OPERATOR_OVERLOAD_IMPL

#define MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD_IMPL(op, op_tag)                                             \
void Variable::operator op(Variable rhs) const noexcept {                                                   \
    void_(Function::current().add_expression(std::make_unique<BinaryExpr>(BinaryOp::op_tag, *this, rhs)));  \
}                                                                                                           \

#define MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD(op_and_tag) \
MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD_IMPL op_and_tag

LUISA_MAP_MACRO(MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD,
                (=, ASSIGN), (+=, ADD_ASSIGN), (-=, SUB_ASSIGN), (*=, MUL_ASSIGN), (/=, DIV_ASSIGN), (%=, MOD_ASSIGN),
                (&=, BIT_AND_ASSIGN), (|=, BIT_OR_ASSIGN), (^=, BIT_XOR_ASSIGN),
                (<<=, SHL_ASSIGN), (>>=, SHR_ASSIGN))

#undef MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD
#undef MAKE_VARIABLE_ASSIGN_OPERATOR_OVERLOAD_IMPL

Variable Variable::member(std::string m) const noexcept { return Function::current().add_expression(std::make_unique<MemberExpr>(*this, std::move(m))); }

Variable Variable::operator+() const noexcept { return Function::current().add_expression(std::make_unique<UnaryExpr>(UnaryOp::PLUS, *this)); }
Variable Variable::operator-() const noexcept { return Function::current().add_expression(std::make_unique<UnaryExpr>(UnaryOp::MINUS, *this)); }
Variable Variable::operator~() const noexcept { return Function::current().add_expression(std::make_unique<UnaryExpr>(UnaryOp::BIT_NOT, *this)); }
Variable Variable::operator!() const noexcept { return Function::current().add_expression(std::make_unique<UnaryExpr>(UnaryOp::NOT, *this)); }
Variable Variable::operator*() const noexcept { return Function::current().add_expression(std::make_unique<UnaryExpr>(UnaryOp::DEREFERENCE, *this)); }
Variable Variable::operator&() const noexcept { return Function::current().add_expression(std::make_unique<UnaryExpr>(UnaryOp::ADDRESS_OF, *this)); }
    
}
