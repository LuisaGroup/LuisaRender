//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <memory>
#include <vector>

namespace luisa {
class Variable;
class Statement;
}

namespace luisa {

namespace detail {

template<typename F>
constexpr auto to_std_function(F &&f) noexcept {
    return std::function{f};
}

template<typename F>
struct FunctionArgumentsImpl {};

template<typename R, typename ...Args>
struct FunctionArgumentsImpl<std::function<R(Args...)>> {
    using Type = std::tuple<Args...>;
};

}

template<typename F>
using FunctionArguments = typename detail::FunctionArgumentsImpl<decltype(detail::to_std_function(std::declval<F>()))>::Type;

class Function {

private:
    std::vector<std::unique_ptr<Variable>> _variables;
    std::vector<uint32_t> _argument_indices;
    std::vector<std::unique_ptr<Statement>> _statements;

public:
    
    virtual ~Function() noexcept = default;
};

}
