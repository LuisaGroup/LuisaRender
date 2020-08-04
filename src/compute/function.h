//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <memory>
#include <set>

#include <compute/statement.h>

namespace luisa::dsl {

namespace detail {

struct TypeDescCmp {
    bool operator()(const TypeDesc *lhs, const TypeDesc *rhs) const noexcept {
        return lhs->uid < rhs->uid;
    }
};

}

class Function {
    
    friend class Variable;

private:
    std::string _name;
    std::vector<Variable> _arguments;
    std::vector<Variable> _builtin_vars;
    std::vector<std::unique_ptr<Expression>> _expressions;
    std::vector<std::unique_ptr<Statement>> _statements;
    std::set<const TypeDesc *, detail::TypeDescCmp> _used_structs;
    
    uint32_t _used_builtins{0u};
    uint32_t _uid_counter{0u};
    
    inline static thread_local Function *_current{nullptr};
    
    [[nodiscard]] uint32_t _get_uid() noexcept { return _uid_counter++; }
    
    template<typename T>
    [[nodiscard]] Variable _builtin_var(BuiltinVariable tag) noexcept {
        auto type = type_desc<T>;
        _use_type(type);
        Variable v{type, tag};
        if (auto bit = static_cast<uint32_t>(tag); (_used_builtins & bit) == 0u) {
            _used_builtins |= bit;
            _builtin_vars.emplace_back(v);
        }
        return v;
    }
    
    void _use_type(const TypeDesc *desc) noexcept {
        if (desc == nullptr || _used_structs.find(desc) != _used_structs.end()) { return; }
        if (desc->type == TypeCatalog::STRUCTURE) {
            _used_structs.emplace(desc);
            for (auto member : desc->member_types) { _use_type(member); }
        } else {
            _use_type(desc->element_type);
        }
    }
    
    template<typename T, bool is_const, typename ...Literals>
    [[nodiscard]] Variable _var_or_const(Literals &&...vs) noexcept {
        auto type = type_desc<T>;
        _use_type(type);
        Variable v{type, _get_uid()};
        add_statement(std::make_unique<DeclareStmt>(v, literal(std::forward<Literals>(vs)...), is_const));
        return v;
    }

public:
    explicit Function(std::string name) noexcept: _name{std::move(name)} { _current = this; }
    ~Function() noexcept { _current = nullptr; }
    
    [[nodiscard]] static Function &current() noexcept {
        assert(_current != nullptr);
        return *_current;
    }
    
    [[nodiscard]] const std::string &name() const noexcept { return _name; }
    
    template<typename T>
    [[nodiscard]] Variable arg() noexcept {
        auto type = type_desc<T>;
        _use_type(type);
        return _arguments.emplace_back(type, _get_uid());
    }
    
    template<typename ...Literals,
        std::enable_if_t<std::conjunction_v<std::is_convertible<Literals, LiteralExpr::Value>...>, int> = 0>
    [[nodiscard]] Variable literal(Literals &&...vs) noexcept {
        std::vector<LiteralExpr::Value> values{std::forward<Literals>(vs)...};
        return add_expression(std::make_unique<LiteralExpr>(std::move(values)));
    }
    
    template<typename Container,
        std::enable_if_t<
            std::conjunction_v<
                std::is_convertible<decltype(*std::cbegin(std::declval<Container>())), LiteralExpr::Value>,
                std::is_convertible<decltype(*std::cend(std::declval<Container>())), LiteralExpr::Value>>, int> = 0>
    [[nodiscard]] Variable literal(Container &&container) noexcept {
        std::vector<LiteralExpr::Value> values{std::begin(container), std::end(container)};
        return add_expression(std::make_unique<LiteralExpr>(this, std::move(values)));
    }
    
    [[nodiscard]] Variable thread_id() noexcept { return _builtin_var<uint32_t>(BuiltinVariable::THREAD_ID); }
    
    template<typename T, typename ...Literals>
    [[nodiscard]] Variable var(Literals &&...vs) noexcept { return _var_or_const<T, false>(std::forward<Literals>(vs)...); }
    
    [[nodiscard]] Variable var(LiteralExpr::Value v) noexcept { return _var_or_const<AutoType, false>(std::move(v)); }
    
    template<typename T, typename ...Literals>
    [[nodiscard]] Variable constant(Literals &&...vs) noexcept { return _var_or_const<T, true>(std::forward<Literals>(vs)...); }
    
    [[nodiscard]] Variable constant(LiteralExpr::Value v) noexcept { return _var_or_const<AutoType, true>(std::move(v)); }
    
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
    [[nodiscard]] const std::vector<Variable> &builtin_variables() const noexcept { return _builtin_vars; }
    [[nodiscard]] const std::vector<std::unique_ptr<Statement>> &statements() const noexcept { return _statements; }
    [[nodiscard]] const auto &used_structures() const noexcept { return _used_structs; }
};

#define LUISA_FUNC        [&](Function &f)
#define LUISA_LAMBDA(...) [&](__VA_ARGS__)

}
