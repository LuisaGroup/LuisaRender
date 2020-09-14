//
// Created by Mike Smith on 2020/9/14.
//

#pragma once

#include <compute/kernel.h>
#include <compute/dispatcher.h>

namespace luisa::compute {

class Pipeline {

private:
    std::vector<std::function<void(Dispatcher &)>> _stages;

public:
    template<typename Func, std::enable_if_t<std::is_invocable_v<Func, Dispatcher &>, int> = 0>
    Pipeline &operator<<(Func &&func) {
        _stages.emplace_back(std::forward<Func>(func));
        return *this;
    }
    
    template<typename Func, std::enable_if_t<std::is_invocable_v<Func>, int> = 0>
    Pipeline &operator<<(Func &&func) {
        _stages.emplace_back([f = std::forward<Func>(func)](Dispatcher &) { f(); });
        return *this;
    }
    
    template<typename Func, std::enable_if_t<std::is_invocable_v<Func, Pipeline &>, int> = 0>
    Pipeline &operator<<(Func &&func) {
        func(*this);
        return *this;
    }
    
    [[nodiscard]] auto run() {
        return [this](Dispatcher &dispatch) {
            for (auto &&stage : _stages) { dispatch(stage); }
        };
    }
    
};

}
