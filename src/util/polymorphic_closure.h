//
// Created by ChenXin on 2023/5/9.
//

#pragma once

#include <utility>

namespace luisa::render {

class PolymorphicClosure {

private:
    std::any _context;

public:
    virtual ~PolymorphicClosure() noexcept = default;

    template<typename T>
    void bind(T &&ctx) noexcept {
        if (!_context.has_value()) {
            _context = std::forward<T>(ctx);
        } else {
            auto p_data = std::any_cast<T>(&_context);
            assert(p_data != nullptr);
            *p_data = std::forward<T>(ctx);
        }
    }

    template<typename T>
    [[nodiscard]] const T &context() const noexcept {
        auto ctx = std::any_cast<T>(&_context);
        assert(ctx != nullptr);
        return *ctx;
    }

    virtual void before_evaluation() noexcept {
        /* prepare any persistent data within a single the dispatch */
    }

    virtual void after_evaluation() noexcept {
        /* release any persistent data within a single dispatch */
    }
};

template<typename Closure>
class PolymorphicCall {

private:
    UInt _tag;
    luisa::unordered_map<luisa::string, uint> _closure_tags;
    luisa::vector<luisa::unique_ptr<Closure>> _closures;

public:
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] auto empty() const noexcept { return _closure_tags.empty(); }
    [[nodiscard]] auto size() const noexcept { return _closure_tags.size(); }
    [[nodiscard]] auto closure(uint index) const noexcept { return _closures[index].get(); }

    using ClosureCreator = luisa::function<luisa::unique_ptr<Closure>()>;
    using ClosureEvaluator = luisa::function<void(const Closure *)>;

    template<typename T = Closure>
        requires std::derived_from<T, Closure>
    [[nodiscard]] T *collect(
        luisa::string_view identifier,
        const ClosureCreator &f = [] { return luisa::make_unique<Closure>(); }) noexcept {

        auto [iter, first] = _closure_tags.try_emplace(
            identifier, static_cast<uint>(_closures.size()));
        if (first) { _closures.emplace_back(f()); }
        _tag = iter->second;
        auto closure = dynamic_cast<T *>(_closures[iter->second].get());
        assert(closure != nullptr);
        return closure;
    }

    void execute(const ClosureEvaluator &f) const noexcept {
        if (empty()) [[unlikely]] { return; }
        if (size() == 1u) {
            _closures.front()->before_evaluation();
            f(_closures.front().get());
            _closures.front()->after_evaluation();
        } else {
            compute::detail::SwitchStmtBuilder{_tag} % [&] {
                for (auto i = 0u; i < size(); i++) {
                    compute::detail::SwitchCaseStmtBuilder{i} % [&f, this, i] {
                        _closures[i]->before_evaluation();
                        f(_closures[i].get());
                        _closures[i]->after_evaluation();
                    };
                }
                compute::detail::SwitchDefaultStmtBuilder{} % compute::unreachable;
            };
        }
    }
};

}// namespace luisa::render
