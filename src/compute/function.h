//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <map>
#include <memory>

#include <compute/statement.h>

namespace luisa::dsl {

class Function {
    
    friend class Variable;

private:
    std::vector<Variable> _arguments;
    std::vector<std::unique_ptr<Expression>> _expressions;
    std::vector<std::unique_ptr<Statement>> _statements;
    std::vector<const TypeDesc *> _used_structs;
    
    uint32_t _used_builtins{0u};
    uint32_t _uid_counter{0u};
    
    [[nodiscard]] uint32_t _get_uid() noexcept { return _uid_counter++; }

public:
    template<typename T>
    Variable arg() noexcept { return _arguments.emplace_back(this, type_desc<T>, _get_uid()); }
    
    template<typename ...Literals, std::enable_if_t<std::conjunction_v<std::is_convertible<Literals, LiteralExpr::Value>...>, int> = 0>
    Variable literal(Literals &&...vs) noexcept {
        std::vector<LiteralExpr::Value> values{std::forward<Literals>(vs)...};
        return add_expression(std::make_unique<LiteralExpr>(this, std::move(values)));
    }
    
    template<typename ...Literals>
    Variable $(Literals &&...vs) noexcept { return literal(std::forward<Literals>(vs)...); }
    
    template<typename T>
    void use() noexcept {
        auto desc = type_desc<T>;
        if (desc->type == TypeCatalog::STRUCTURE) {
            _used_structs.emplace_back(desc);
        } else {
            LUISA_WARNING("Type \"", to_string(desc), "\" is not a user-defined structure, usage ignored.");
        }
    }
    
    Variable thread_id() noexcept {
        Variable tid{this, type_desc<uint32_t>, BuiltinTag::THREAD_ID};
        if (auto bit = static_cast<uint32_t>(BuiltinTag::THREAD_ID); (_used_builtins & bit) == 0u) {
            _used_builtins |= bit;
            _arguments.emplace_back(tid);
        }
        return tid;
    }
    
    template<typename T>
    Variable var(Variable init) noexcept {
        Variable v{this, type_desc<T>, _get_uid()};
        add_statement(std::make_unique<DeclareStmt>(v, init));
        return v;
    }
    
    template<typename T, typename ...Literals, typename std::enable_if_t<std::is_constructible_v<T, Literals...>, int> = 0>
    Variable var(Literals &&...vs) noexcept {
        return var<T>(literal(std::forward<Literals>(vs)...));
    }
    
    [[nodiscard]] Variable add_expression(std::unique_ptr<Expression> expr) noexcept {
        return Variable{_expressions.emplace_back(std::move(expr)).get()};
    }
    
    void add_statement(std::unique_ptr<Statement> stmt) noexcept {
        _statements.emplace_back(std::move(stmt));
    }
    
    template<typename Block, std::enable_if_t<std::is_invocable_v<Block>, int> = 0>
    void block(Block &&def_block) noexcept {
        add_statement(std::make_unique<KeywordStmt>("{"));
        def_block();
        add_statement(std::make_unique<KeywordStmt>("}"));
    }
    
    void add_break() noexcept { add_statement(std::make_unique<KeywordStmt>("break;")); }
    void add_continue() noexcept { add_statement(std::make_unique<KeywordStmt>("continue;")); }
    void add_return() noexcept { add_statement(std::make_unique<KeywordStmt>("return;")); }
    
    [[nodiscard]] const std::vector<Variable> &arguments() const noexcept { return _arguments; }
    [[nodiscard]] const std::vector<std::unique_ptr<Statement>> &statements() const noexcept { return _statements; }
    [[nodiscard]] const std::vector<const TypeDesc *> &used_structures() const noexcept { return _used_structs; }
};

#define LUISA_FUNC [&](Function &f)

}
