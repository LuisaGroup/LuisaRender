//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <utility>
#include <variant>

#include <compute/variable.h>

namespace luisa::dsl {

struct ExprVisitor;

class Expression {

private:
    Function *_function;

public:
    virtual ~Expression() noexcept = default;
    explicit Expression(Function *func) noexcept: _function{func} {}
    [[nodiscard]] Function *function() const noexcept { return _function; }
    
    virtual void accept(const ExprVisitor &) const = 0;
};

// fwd-decl of derived expressions
class UnaryExpr;
class BinaryExpr;
class MemberExpr;
class ArrowExpr;
class LiteralExpr;
class BuiltinFuncExpr;
class CastExpr;

struct ExprVisitor {
    virtual void visit(const UnaryExpr &unary_expr) const = 0;
    virtual void visit(const BinaryExpr &binary_expr) const = 0;
    virtual void visit(const MemberExpr &member_expr) const = 0;
    virtual void visit(const ArrowExpr &arrow_expr) const = 0;
    virtual void visit(const LiteralExpr &literal_expr) const = 0;
    virtual void visit(const BuiltinFuncExpr &func_expr) const = 0;
    virtual void visit(const CastExpr &cast_expr) const = 0;
};

#define MAKE_EXPRESSION_ACCEPT_VISITOR()                                          \
void accept(const ExprVisitor &visitor) const override { visitor.visit(*this); }  \

enum struct UnaryOp {
    NOT,          // !x
    BIT_NOT,      // ~x
    ADDRESS_OF,   // &x
    DEREFERENCE,  // *x
    PREFIX_INC,   // ++x
    PREFIX_DEC,   // --x
    POSTFIX_INC,  // x++
    POSTFIX_DEC   // x--
};

class UnaryExpr : public Expression {

private:
    Variable _operand;
    UnaryOp _op;

public:
    UnaryExpr(UnaryOp op, Variable operand) noexcept
        : Expression{operand.function()}, _operand{std::move(operand)}, _op{op} {}
    
    [[nodiscard]] Variable operand() const noexcept { return _operand; }
    [[nodiscard]] UnaryOp op() const noexcept { return _op; }
    MAKE_EXPRESSION_ACCEPT_VISITOR()
};

enum struct BinaryOp {
    
    // arithmetic
    ADD, SUB, MUL, DIV, MOD,
    BIT_AND, BIT_OR, BIT_XOR, SHL, SHR,
    AND, OR,
    
    // relational
    LESS, GREATER, LESS_EQUAL, GREATER_EQUAL, EQUAL, NOT_EQUAL,
    
    // operator[]
    ACCESS,
    
    // assignment
    ASSIGN,
    ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, DIV_ASSIGN, MOD_ASSIGN,
    BIT_AND_ASSIGN, BIT_OR_ASSIGN, BIT_XOR_ASSIGN, SHL_ASSIGN, SHR_ASSIGN
};

class BinaryExpr : public Expression {

private:
    Variable _lhs;
    Variable _rhs;
    BinaryOp _op;

public:
    BinaryExpr(BinaryOp op, Variable lhs, Variable rhs) noexcept
        : Expression{lhs.function()}, _op{op}, _lhs{std::move(lhs)}, _rhs{std::move(rhs)} {}
    
    [[nodiscard]] Variable lhs() const noexcept { return _lhs; }
    [[nodiscard]] Variable rhs() const noexcept { return _rhs; }
    [[nodiscard]] BinaryOp op() const noexcept { return _op; }
    MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class MemberExpr : public Expression {

private:
    Variable _self;
    std::string _member;

public:
    MemberExpr(Variable self, std::string member) noexcept: Expression{self.function()}, _self{self}, _member{std::move(member)} {}
    [[nodiscard]] Variable self() const noexcept { return _self; }
    [[nodiscard]] const std::string &member() const noexcept { return _member; }
    MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class ArrowExpr : public Expression {

private:
    Variable _self;
    std::string _member;

public:
    ArrowExpr(Variable self, std::string member) noexcept: Expression{self.function()}, _self{self}, _member{std::move(member)} {}
    [[nodiscard]] Variable self() const noexcept { return _self; }
    [[nodiscard]] const std::string &member() const noexcept { return _member; }
    MAKE_EXPRESSION_ACCEPT_VISITOR()
};

class LiteralExpr : public Expression {

public:
    using Value = std::variant<bool, float, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;

private:
    std::vector<Value> _values;

public:
    LiteralExpr(Function *f, std::vector<Value> values) noexcept
        : Expression{f}, _values{std::move(values)} {}
    
    [[nodiscard]] const std::vector<Value> &values() const noexcept { return _values; }
    MAKE_EXPRESSION_ACCEPT_VISITOR()
};

enum struct BuiltinFunc {
    
    SELECT,
    
    SIN, COS, TAN, ASIN, ACOS, ATAN, ATAN2,
    CEIL, FLOOR, ROUND,
    POW, EXP, LOG, LOG2, LOG10,
    MIN, MAX,
    ABS, CLAMP, LERP,
    
    RADIANS, DEGREES,
    
    NORMALIZE, LENGTH, DOT, CROSS,
    MAKE_MAT3, MAKE_MAT4, INVERSE, TRANSPOSE,

#define BUILTIN_FUNC_MAKE_VECTOR(T)  \
    MAKE_##T##2, MAKE_##T##3, MAKE_##T##4, MAKE_PACKED_##T##3
    
    BUILTIN_FUNC_MAKE_VECTOR(FLOAT),
    BUILTIN_FUNC_MAKE_VECTOR(BOOL),
    BUILTIN_FUNC_MAKE_VECTOR(BYTE), BUILTIN_FUNC_MAKE_VECTOR(UBYTE),
    BUILTIN_FUNC_MAKE_VECTOR(SHORT), BUILTIN_FUNC_MAKE_VECTOR(USHORT),
    BUILTIN_FUNC_MAKE_VECTOR(INT), BUILTIN_FUNC_MAKE_VECTOR(UINT),
    BUILTIN_FUNC_MAKE_VECTOR(LONG), BUILTIN_FUNC_MAKE_VECTOR(ULONG),

#undef BUILTIN_FUNC_MAKE_VECTOR
    
};

class BuiltinFuncExpr : public Expression {

private:
    std::vector<Variable> _arguments;
    BuiltinFunc _func;

public:
    BuiltinFuncExpr(BuiltinFunc func, std::vector<Variable> args) noexcept
        : Expression{args.front().function()}, _func{func}, _arguments{std::move(args)} {}
    
    [[nodiscard]] BuiltinFunc builtin_function() const noexcept { return _func; }
    [[nodiscard]] const std::vector<Variable> &arguments() const noexcept { return _arguments; }
    MAKE_EXPRESSION_ACCEPT_VISITOR()
};

enum struct CastOp {
    STATIC,
    REINTERPRET,
    BITWISE
};

class CastExpr : public Expression {

private:
    Variable _source;
    CastOp _op;
    const TypeDesc *_dest_type;

public:
    CastExpr(CastOp op, Variable src, const TypeDesc *dest) noexcept
        : Expression{src.function()}, _op{op}, _source{std::move(src)}, _dest_type{dest} {}
    [[nodiscard]] CastOp op() const noexcept { return _op; }
    [[nodiscard]] Variable source() const noexcept { return _source; }
    [[nodiscard]] const TypeDesc *dest_type() const noexcept { return _dest_type; }
    MAKE_EXPRESSION_ACCEPT_VISITOR()
};

#undef MAKE_EXPRESSION_ACCEPT_VISITOR

}
