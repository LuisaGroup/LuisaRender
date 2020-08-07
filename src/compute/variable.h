//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <variant>
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

namespace detail {

template<typename T>
using EnableIfLiteralOperand = std::enable_if_t<
    std::is_convertible_v<T, std::variant<
        bool, float, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>>, int>;
    
}

class Variable {

protected:
    // For variable declarations
    const TypeDesc *_type{nullptr};
    uint32_t _uid{0u};
    
    bool _is_argument{false};
    
    // For builtin variables
    BuiltinVariable _builtin_tag{BuiltinVariable::NOT_BUILTIN};
    
    // For temporary variables in expressions
    Expression *_expression{nullptr};

public:
    Variable(const TypeDesc *type, uint32_t uid, bool is_argument = false) noexcept;
    Variable(const TypeDesc *type, BuiltinVariable tag) noexcept;
    explicit Variable(Expression *expr) noexcept;
    Variable(Variable &&) = default;
    Variable(const Variable &) = default;
    
    [[nodiscard]] bool is_argument() const noexcept { return _is_argument; }
    [[nodiscard]] Expression *expression() const noexcept { return _expression; }
    [[nodiscard]] const TypeDesc *type() const noexcept { return _type; }
    [[nodiscard]] uint32_t uid() const noexcept { return _uid; }
    [[nodiscard]] BuiltinVariable builtin_tag() const noexcept { return _builtin_tag; }
    [[nodiscard]] bool is_temporary() const noexcept { return _expression != nullptr; }
    [[nodiscard]] bool is_builtin() const noexcept { return _builtin_tag != BuiltinVariable::NOT_BUILTIN; }
    
    [[nodiscard]] Variable member(std::string m) const noexcept;
    [[nodiscard]] Variable operator[](std::string m) const noexcept { return member(std::move(m)); }
    
    // Convenient methods for accessing vector members
    [[nodiscard]] Variable x() const noexcept { return member("x"); }
    [[nodiscard]] Variable y() const noexcept { return member("y"); }
    [[nodiscard]] Variable z() const noexcept { return member("z"); }
    [[nodiscard]] Variable w() const noexcept { return member("w"); }
    [[nodiscard]] Variable r() const noexcept { return member("r"); }
    [[nodiscard]] Variable g() const noexcept { return member("g"); }
    [[nodiscard]] Variable b() const noexcept { return member("b"); }
    [[nodiscard]] Variable a() const noexcept { return member("a"); }
    
    [[nodiscard]] Variable operator+() const noexcept;
    [[nodiscard]] Variable operator-() const noexcept;
    [[nodiscard]] Variable operator~() const noexcept;
    [[nodiscard]] Variable operator!() const noexcept;
    [[nodiscard]] Variable operator*() const noexcept;
    [[nodiscard]] Variable operator&() const noexcept;

#define MAKE_BINARY_OPERATOR_DECL(op)                                                                                    \
[[nodiscard]] Variable operator op(Variable rhs) const noexcept;                                                         \
template<typename T, detail::EnableIfLiteralOperand<T> = 0> [[nodiscard]] Variable operator op(T &&rhs) const noexcept;  \

#define MAKE_ASSIGNMENT_OPERATOR_DECL(op)                                                                                \
void operator op(Variable rhs) const noexcept;                                                                           \
template<typename T, detail::EnableIfLiteralOperand<T> = 0> void operator op(T &&rhs) const noexcept;                    \

    LUISA_MAP_MACRO(MAKE_BINARY_OPERATOR_DECL, +, -, *, /, %, <<, >>, &, |, ^, &&, ||, ==, !=, <,>, <=, >=, [])
    LUISA_MAP_MACRO(MAKE_ASSIGNMENT_OPERATOR_DECL, =, +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=)

#undef MAKE_BINARY_OPERATOR_DECL
#undef MAKE_ASSIGNMENT_OPERATOR_DECL

};

#define $(m) member(#m)

}
