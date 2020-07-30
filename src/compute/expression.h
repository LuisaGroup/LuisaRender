//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <utility>
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

struct ExprVisitor {
    virtual void visit(const UnaryExpr &unary_expr) const = 0;
    virtual void visit(const BinaryExpr &binary_expr) const = 0;
    virtual void visit(const MemberExpr &member_expr) const = 0;
    virtual void visit(const ArrowExpr &arrow_expr) const = 0;
    virtual void visit(const LiteralExpr &literal_expr) const = 0;
    virtual void visit(const BuiltinFuncExpr &func_expr) const = 0;
};

#define MAKE_EXPRESSION_ACCEPT_VISITOR()                                          \
void accept(const ExprVisitor &visitor) const override { visitor.visit(*this); }  \

enum struct UnaryOp {
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

private:
    std::string _literal;

public:
    LiteralExpr(Function *function, std::string literal) noexcept
        : Expression{function}, _literal{std::move(literal)} {}
    
    [[nodiscard]] const std::string &literal() const noexcept { return _literal; }
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

#define MAKE_VECTOR(T)  \
    MAKE_##T##2, MAKE_##T##3, MAKE_##T##4, MAKE_PACKED_##T##3
    
    MAKE_VECTOR(FLOAT),
    MAKE_VECTOR(BOOL),
    MAKE_VECTOR(BYTE), MAKE_VECTOR(UBYTE),
    MAKE_VECTOR(SHORT), MAKE_VECTOR(USHORT),
    MAKE_VECTOR(INT), MAKE_VECTOR(UINT),
    MAKE_VECTOR(LONG), MAKE_VECTOR(ULONG),

#undef MAKE_VECTOR
    
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

#undef MAKE_EXPRESSION_ACCEPT_VISITOR

// Built-in Function Declarations
#define MAP_VARIABLE_NAME_TO_ARGUMENT_DECL(name) Variable name
#define MAP_VARIABLE_NAMES_TO_ARGUMENT_LIST(...) LUISA_MAP_MACRO_LIST(MAP_VARIABLE_NAME_TO_ARGUMENT_DECL, __VA_ARGS__)

#define MAKE_BUILTIN_FUNCTION_DECL(func, ...)                                   \
[[nodiscard]] Variable func(MAP_VARIABLE_NAMES_TO_ARGUMENT_LIST(__VA_ARGS__));  \

MAKE_BUILTIN_FUNCTION_DECL(select_, cond, tv, fv)

MAKE_BUILTIN_FUNCTION_DECL(sin_, x)
MAKE_BUILTIN_FUNCTION_DECL(cos_, x)
MAKE_BUILTIN_FUNCTION_DECL(tan_, x)
MAKE_BUILTIN_FUNCTION_DECL(asin_, x)
MAKE_BUILTIN_FUNCTION_DECL(acos_, x)
MAKE_BUILTIN_FUNCTION_DECL(atan_, x)
MAKE_BUILTIN_FUNCTION_DECL(atan2_, y, x)
MAKE_BUILTIN_FUNCTION_DECL(ceil_, x)
MAKE_BUILTIN_FUNCTION_DECL(floor_, x)
MAKE_BUILTIN_FUNCTION_DECL(round_, x)
MAKE_BUILTIN_FUNCTION_DECL(pow_, x)
MAKE_BUILTIN_FUNCTION_DECL(exp_, x)
MAKE_BUILTIN_FUNCTION_DECL(log_, x)
MAKE_BUILTIN_FUNCTION_DECL(log2_, x)
MAKE_BUILTIN_FUNCTION_DECL(log10_, x)
MAKE_BUILTIN_FUNCTION_DECL(min_, x, y)
MAKE_BUILTIN_FUNCTION_DECL(max_, x, y)
MAKE_BUILTIN_FUNCTION_DECL(abs_, x)
MAKE_BUILTIN_FUNCTION_DECL(clamp_, x, a, b)
MAKE_BUILTIN_FUNCTION_DECL(lerp_, a, b, t)
MAKE_BUILTIN_FUNCTION_DECL(radians_, deg)
MAKE_BUILTIN_FUNCTION_DECL(degrees_, rad)
MAKE_BUILTIN_FUNCTION_DECL(normalize_, v)
MAKE_BUILTIN_FUNCTION_DECL(length_, v)
MAKE_BUILTIN_FUNCTION_DECL(dot_, u, v)
MAKE_BUILTIN_FUNCTION_DECL(cross_, u, v)

MAKE_BUILTIN_FUNCTION_DECL(make_mat3_, val_or_mat4)
MAKE_BUILTIN_FUNCTION_DECL(make_mat3_, c0, c1, c2)
MAKE_BUILTIN_FUNCTION_DECL(make_mat3_, m00, m01, m02, m10, m11, m12, m20, m21, m22)

MAKE_BUILTIN_FUNCTION_DECL(make_mat4_, val_or_mat3)
MAKE_BUILTIN_FUNCTION_DECL(make_mat4_, c0, c1, c2, c3)
MAKE_BUILTIN_FUNCTION_DECL(make_mat4_, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33)

// make_vec2
#define MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC2(T)       \
MAKE_BUILTIN_FUNCTION_DECL(make_##T##2_, v)           \
MAKE_BUILTIN_FUNCTION_DECL(make_##T##2_, x, y)        \

// make_vec3
#define MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC3(T)       \
MAKE_BUILTIN_FUNCTION_DECL(make_##T##3_, v)           \
MAKE_BUILTIN_FUNCTION_DECL(make_##T##3_, x, y)        \
MAKE_BUILTIN_FUNCTION_DECL(make_##T##3_, x, y, z)     \

// make_vec4
#define MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC4(T)       \
MAKE_BUILTIN_FUNCTION_DECL(make_##T##4_, v)           \
MAKE_BUILTIN_FUNCTION_DECL(make_##T##4_, x, y)        \
MAKE_BUILTIN_FUNCTION_DECL(make_##T##4_, x, y, z)     \
MAKE_BUILTIN_FUNCTION_DECL(make_##T##4_, x, y, z, w)  \

#define MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(T)        \
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC2(T)               \
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC3(T)               \
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC4(T)               \

MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(bool)
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(float)
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(byte)
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(ubyte)
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(short)
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(ushort)
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(int)
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(uint)
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(long)
MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC(ulong)

#undef MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC
#undef MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC4
#undef MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC3
#undef MAKE_BUILTIN_FUNCTION_DECL_MAKE_VEC2

#undef MAKE_BUILTIN_FUNCTION_DECL
#undef MAP_VARIABLE_NAMES_TO_ARGUMENT_LIST
#undef MAP_VARIABLE_NAME_TO_ARGUMENT_DECL

}
