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
        for (auto i = 0u; i < _stages.size(); i += 16u) {
            _device->launch([this, i](Dispatcher &dispatch) {
                for (auto j = 0u; j < 16u && i + j < _stages.size(); j++) {
                    dispatch(_stages[i + j]);
                }
            });
        }
    }
    
};

}
