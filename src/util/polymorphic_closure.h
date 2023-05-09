//
// Created by ChenXin on 2023/5/9.
//

#pragma once

#include <utility>

namespace luisa::render {

template<typename BaseFunction>
class PolymorphicClosure {

public:
    using Self = PolymorphicClosure<BaseFunction>;

    class Context {

    private:
        std::any _data;
        luisa::unordered_map<luisa::string, luisa::unique_ptr<Self>> _nested;

    public:
        // for creation
        template<typename Data>
        void create_data(Data &&data) noexcept {
            if (!_data.has_value()) {
                _data = std::forward<Data>(data);
            } else {
                auto p_data = std::any_cast<Data>(&_data);
                assert(p_data != nullptr);
                *p_data = std::forward<Data>(data);
            }
        }

        [[nodiscard]] auto &create_nested(luisa::string name) noexcept {
            auto [iter, _] = _nested.try_emplace(
                std::move(name),
                luisa::lazy_construct([] { return luisa::make_unique<Self>(); }));
            return *iter->second;
        }

        // for dispatching
        template<typename Data>
        [[nodiscard]] auto data() const noexcept {
            return std::any_cast<Data>(&_data);
        }

        [[nodiscard]] auto &nested(luisa::string_view name) const noexcept {
            return *_nested.at(name);
        }
    };

private:
    luisa::unordered_map<luisa::string, uint> _tags;
    luisa::vector<luisa::unique_ptr<BaseFunction>> _functions;
    luisa::vector<luisa::unique_ptr<Context>> _ctx_slots;

public:
    [[nodiscard]] auto empty() const noexcept { return _tags.empty(); }
    [[nodiscard]] auto size() const noexcept { return _tags.size(); }
    [[nodiscard]] auto function(uint index) const noexcept { return _functions[index].get(); }
    [[nodiscard]] auto context(uint index) const noexcept { return _ctx_slots[index].get(); }

    template<typename Function>
        requires std::derived_from<Function, BaseFunction>
    [[nodiscard]] std::pair<uint, Context *> register_instance(luisa::string_view class_identifier) noexcept {
        auto [iter, first] = _tags.try_emplace(class_identifier, _tags.size());
        auto tag = iter->second;
        if (first) {
            _functions.emplace_back(luisa::make_unique<Function>());
            _ctx_slots.emplace_back(luisa::make_unique<Context>());
        }
        return std::make_pair(tag, _ctx_slots[tag].get());
    }

    template<typename Tag>
        requires compute::is_integral_expr_v<Tag>
    void dispatch(Tag &&tag, const luisa::function<void(const BaseFunction *, const Context *)> &f) const noexcept {
        if (empty()) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION("No implementations registered.");
        }
        if (_tags.size() == 1u) {
            f(_functions.front().get(), _ctx_slots.front().get());
        } else {
            compute::detail::SwitchStmtBuilder{std::forward<Tag>(tag)} % [&] {
                for (auto i = 0u; i < _tags.size(); i++) {
                    compute::detail::SwitchCaseStmtBuilder{i} % [&f, this, i] {
                        f(_functions[i].get(), _ctx_slots[i].get());
                    };
                }
                compute::detail::SwitchDefaultStmtBuilder{} % compute::unreachable;
            };
        }
    }
};

}