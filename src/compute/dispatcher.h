//
// Created by Mike Smith on 2020/8/9.
//

#pragma once

#include <vector>
#include <thread>
#include <future>
#include <functional>
#include <core/concepts.h>

namespace luisa::compute {

class Dispatcher : Noncopyable {

protected:
    std::vector<std::function<void()>> _callbacks;
    std::future<void> _future;
    virtual void _synchronize() = 0;
    virtual void _commit() = 0;

public:
    virtual ~Dispatcher() noexcept = default;
    
    template<typename Callback, std::enable_if_t<std::is_invocable_v<Callback>, int> = 0>
    void add_callback(Callback &&f) noexcept { _callbacks.emplace_back(std::forward<Callback>(f)); }
    
    virtual void commit() {
        _commit();
        _future = std::async(std::launch::async, [this] {
            _synchronize();
            for (auto &&cb : _callbacks) { cb(); }
        });
    }
    
    virtual void synchronize() { if (_future.valid()) { _future.wait(); }}
};

}
