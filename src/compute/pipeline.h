//
// Created by Mike Smith on 2020/9/14.
//

#pragma once

#include <compute/device.h>
#include <compute/kernel.h>
#include <compute/dispatcher.h>

namespace luisa::compute {

class Pipeline : public Noncopyable {

private:
    Device *_device;
    std::vector<std::function<void(Dispatcher &)>> _stages;

public:
    explicit Pipeline(Device *device) : _device{device} {}
    
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
    
    void run() {
        _device->launch([this](Dispatcher &dispatch) {
            for (auto &&stage : _stages) { stage(dispatch); }
        });
    }
    
};

}
