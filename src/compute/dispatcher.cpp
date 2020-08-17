//
// Created by Mike Smith on 2020/8/17.
//

#include "dispatcher.h"
#include "kernel.h"

namespace luisa::compute {

void Dispatcher::_commit() {
    _do_commit();
    _future = std::async(std::launch::async, [this] {
        _do_synchronize();
        for (auto &&cb : _callbacks) { cb(); }
    });
}

void Dispatcher::_synchronize() { if (_future.valid()) { _future.wait(); }}

void Dispatcher::operator()(Kernel &kernel, uint threads, uint tg_size) { kernel.dispatch(*this, threads, tg_size); }
void Dispatcher::operator()(Kernel &kernel, uint2 threads, uint2 tg_size) { kernel.dispatch(*this, threads, tg_size); }
void Dispatcher::operator()(Kernel &kernel, uint3 threads, uint3 tg_size) { kernel.dispatch(*this, threads, tg_size); }

}
