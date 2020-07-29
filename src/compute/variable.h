//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <compute/type_desc.h>

namespace luisa::dsl {
class Expression;
class Function;
}

enum struct BuiltinTag : uint32_t {
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
    BuiltinTag _builtin_tag{BuiltinTag::NOT_BUILTIN};
    
    // For temporary variables in expressions
    Expression *_expression{nullptr};

public:
    Variable(Function *func, const TypeDesc *type, uint32_t uid) noexcept;
    Variable(Function *func, const TypeDesc *type, BuiltinTag tag) noexcept;
    explicit Variable(Expression *expr) noexcept;
    Variable(Variable &&) = default;
    Variable(const Variable &) = default;
    
    [[nodiscard]] Function *function() const noexcept { return _function; }
    [[nodiscard]] Expression *expression() const noexcept { return _expression; }
    [[nodiscard]] const TypeDesc *type() const noexcept { return _type; }
    [[nodiscard]] uint32_t uid() const noexcept { return _uid; }
    [[nodiscard]] BuiltinTag builtin_tag() const noexcept { return _builtin_tag; }
    [[nodiscard]] bool is_temporary() const noexcept { return _expression != nullptr; }
    [[nodiscard]] bool is_builtin() const noexcept { return _builtin_tag != BuiltinTag::NOT_BUILTIN; }
    
    [[nodiscard]] Variable operator+ (Variable rhs) const noexcept;
    [[nodiscard]] Variable operator- (Variable rhs) const noexcept;
    [[nodiscard]] Variable operator* (Variable rhs) const noexcept;
    [[nodiscard]] Variable operator/ (Variable rhs) const noexcept;
    [[nodiscard]] Variable operator% (Variable rhs) const noexcept;
    [[nodiscard]] Variable operator<<(Variable rhs) const noexcept;
    [[nodiscard]] Variable operator>>(Variable rhs) const noexcept;
    [[nodiscard]] Variable operator& (Variable rhs) const noexcept;
    [[nodiscard]] Variable operator| (Variable rhs) const noexcept;
    [[nodiscard]] Variable operator^ (Variable rhs) const noexcept;
    [[nodiscard]] Variable operator&&(Variable rhs) const noexcept;
    [[nodiscard]] Variable operator||(Variable rhs) const noexcept;
    [[nodiscard]] Variable operator==(Variable rhs) const noexcept;
    [[nodiscard]] Variable operator!=(Variable rhs) const noexcept;
    [[nodiscard]] Variable operator<(Variable rhs) const noexcept;
    [[nodiscard]] Variable operator>(Variable rhs) const noexcept;
    [[nodiscard]] Variable operator<=(Variable rhs) const noexcept;
    [[nodiscard]] Variable operator>=(Variable rhs) const noexcept;
    [[nodiscard]] Variable operator[](Variable rhs) const noexcept;
    
    void operator=(Variable rhs) const noexcept;
};

}
