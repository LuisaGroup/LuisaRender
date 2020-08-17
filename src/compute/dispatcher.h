//
// Created by Mike Smith on 2020/8/9.
//

#pragma once

#include <vector>
#include <thread>
#include <future>
#include <functional>
#include <core/concepts.h>
#include <core/data_types.h>

namespace luisa::compute {

class Device;
class Kernel;

class Dispatcher : Noncopyable {

public:
    friend class Device;

protected:
    std::vector<std::function<void()>> _callbacks;
    std::future<void> _future;
    
    virtual void _do_synchronize() = 0;
    virtual void _do_commit() = 0;
    
    void _commit();
    void _synchronize();

public:
    virtual ~Dispatcher() noexcept = default;
    
    template<typename Callback, std::enable_if_t<std::is_invocable_v<Callback>, int> = 0>
    void add_callback(Callback &&f) noexcept { _callbacks.emplace_back(std::forward<Callback>(f)); }
    
    template<typename F, std::enable_if_t<std::is_invocable_v<F, Dispatcher &>, int> = 0>
    void operator()(F &&f) { f(*this); }
    
    void operator()(Kernel &kernel, uint threads, uint tg_size = 128u);
    void operator()(Kernel &kernel, uint2 threads, uint2 tg_size = make_uint2(8u, 8u));
    void operator()(Kernel &kernel, uint3 threads, uint3 tg_size);
};

}
