//
// Created by Mike Smith on 2020/8/17.
//

#include "dispatcher.h"
#include "kernel.h"

namespace luisa::compute {

void Dispatcher::_commit() {
    _schedule();
    _future = std::async(std::launch::async, [this] {
        _wait();
        std::for_each(_callbacks.cbegin(), _callbacks.cend(), [](auto &&cb) { cb(); });
    });
}

void Dispatcher::_synchronize() { if (_future.valid()) { _future.wait(); }}

void Dispatcher::operator()(Kernel &kernel, uint threads, uint tg_size) { kernel.dispatch(*this, threads, tg_size); }
void Dispatcher::operator()(Kernel &kernel, uint2 threads, uint2 tg_size) { kernel.dispatch(*this, threads, tg_size); }

}
