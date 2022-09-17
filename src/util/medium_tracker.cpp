//
// Created by Mike Smith on 2022/2/19.
//

#include <util/medium_tracker.h>

namespace luisa::render {

using namespace luisa::compute;

MediumTracker::MediumTracker() noexcept : _size{0u} {
    for (auto i = 0u; i < capacity; i++) {
        _priority_list[i] = 0u;
    }
}

Bool MediumTracker::true_hit(Expr<uint> priority) const noexcept {
    return priority > _priority_list[0u];
}

void MediumTracker::enter(Expr<uint> priority, Expr<MediumInfo> value) noexcept {
    auto x = def(priority);
    auto v = def(value);
    for (auto i = 0u; i < capacity; i++) {
        auto p = _priority_list[i];
        auto m = _medium_list[i];
        auto should_swap = p <= x;
        _priority_list[i] = ite(should_swap, x, p);
        _medium_list[i] = ite(should_swap, v, m);
        x = ite(should_swap, p, x);
        v = ite(should_swap, m, v);
    }
}

void MediumTracker::exit(Expr<uint> priority) noexcept {
    for (auto i = 0u; i < capacity - 1u; i++) {
        auto p = _priority_list[i];
        auto should_move = p <= priority;
        auto index = ite(should_move, i + 1u, i);
        _priority_list[i] = _priority_list[index];
        _medium_list[i] = _medium_list[index];
    }
    _priority_list[capacity - 1u] = 0u;
}

Var<MediumInfo> MediumTracker::current() const noexcept {
    auto m = def<MediumInfo>();
    m.eta = ite(vacuum(), 1.f, _medium_list[0].eta);
    return m;
}

Bool MediumTracker::vacuum() const noexcept {
    return _priority_list[0] == 0u;
}

}// namespace luisa::render
