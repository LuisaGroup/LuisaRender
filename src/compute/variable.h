//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <compute/type_desc.h>

namespace luisa::dsl {
class Expression;
class Function;
}

enum struct BuiltinVariable : uint32_t {
    NOT_BUILTIN = 0u,
    THREAD_ID = 1u,
};

namespace luisa::dsl {

class Variable {

private:
    Function *_function{nullptr};
    
    // For variable declarations
    const TypeDesc *_type{nullptr};
    uint32_t _uid{0u};
    
    // For builtin variables
    BuiltinVariable _builtin_tag{BuiltinVariable::NOT_BUILTIN};
    
    // For temporary variables in expressions
    Expression *_expression{nullptr};

public:
    Variable(Function *func, const TypeDesc *type, uint32_t uid) noexcept;
    Variable(Function *func, const TypeDesc *type, BuiltinVariable tag) noexcept;
    explicit Variable(Expression *expr) noexcept;
    Variable(Variable &&) = default;
    Variable(const Variable &) = default;
    
    [[nodiscard]] Function *function() const noexcept { return _function; }
    [[nodiscard]] Expression *expression() const noexcept { return _expression; }
    [[nodiscard]] const TypeDesc *type() const noexcept { return _type; }
    [[nodiscard]] uint32_t uid() const noexcept { return _uid; }
    [[nodiscard]] BuiltinVariable builtin_tag() const noexcept { return _builtin_tag; }
    [[nodiscard]] bool is_temporary() const noexcept { return _expression != nullptr; }
    [[nodiscard]] bool is_builtin() const noexcept { return _builtin_tag != BuiltinVariable::NOT_BUILTIN; }
    
    [[nodiscard]] Variable member(std::string m) const noexcept;
    [[nodiscard]] Variable arrow(std::string m) const noexcept;
    [[nodiscard]] Variable $(std::string m) const noexcept { return member(std::move(m)); }
    [[nodiscard]] Variable p$(std::string m) const noexcept { return arrow(std::move(m)); }
    
    [[nodiscard]] Variable operator~() const noexcept;
    [[nodiscard]] Variable operator!() const noexcept;
    [[nodiscard]] Variable operator*() const noexcept;
    [[nodiscard]] Variable operator&() const noexcept;
    [[nodiscard("Use \"void_(++x)\" to add it into function statements.")]] Variable operator++() const noexcept;
    [[nodiscard("Use \"void_(--x)\" to add it into function statements.")]] Variable operator--() const noexcept;
    [[nodiscard("Use \"void_(x++)\" to add it into function statements.")]] Variable operator++(int) const noexcept;
    [[nodiscard("Use \"void_(x--)\" to add it into function statements.")]] Variable operator--(int) const noexcept;

#define MAKE_BINARY_OPERATOR_DECL(op) [[nodiscard]] Variable operator op(Variable rhs) const noexcept;
    LUISA_MAP_MACRO(MAKE_BINARY_OPERATOR_DECL, +, -, *, /, %, <<, >>, &, |, ^, &&, ||, ==, !=, <,>, <=, >=, [])
#undef MAKE_BINARY_OPERATOR_DECL

#define MAKE_ASSIGNMENT_OPERATOR_DECL(op) [[nodiscard("Use \"void_(lhs " #op " rhs)\" to add it into function statements.")]] Variable operator op(Variable rhs) const noexcept;
    LUISA_MAP_MACRO(MAKE_ASSIGNMENT_OPERATOR_DECL, =, +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=)
#undef MAKE_ASSIGNMENT_OPERATOR_DECL

};

}
