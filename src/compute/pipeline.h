//
// Created by Mike Smith on 2020/9/6.
//

#pragma once

#include <compute/device.h>

namespace luisa::compute {

class Pipeline : public Noncopyable {

private:
    std::vector<std::function<void(Dispatcher &)>> _stages;

public:
    template<typename Stage, std::enable_if_t<std::is_invocable_v<Stage, Dispatcher &>, int> = 0>
    Pipeline &operator<<(Stage &&stage) noexcept {
        _stages.emplace_back(std::forward<Stage>(stage));
        return *this;
    }
    
    template<typename Stage, std::enable_if_t<std::is_invocable_v<Stage>, int> = 0>
    Pipeline &operator<<(Stage &&stage) noexcept {
        _stages.emplace_back([stage = std::function{std::forward<Stage>(stage)}](Dispatcher &) { stage(); });
        return *this;
    }
    
    void operator()(Dispatcher &dispatch) const {
        for (auto &&stage : _stages) { dispatch(stage); }
    }
    
};

}
