//
// Created by Mike Smith on 2022/2/19.
//

#include <util/medium_tracker.h>
#include <dsl/sugar.h>

namespace luisa::render {

using namespace luisa::compute;

Bool equal(Expr<MediumInfo> a, Expr<MediumInfo> b) noexcept {
    return a.medium_tag == b.medium_tag;
}

MediumTracker::MediumTracker() noexcept : _size{0u} {
    for (auto i = 0u; i < capacity; i++) {
        _priority_list[i] = Medium::VACUUM_PRIORITY;
        _medium_list[i] = make_medium_info(Medium::VACUUM_PRIORITY, Medium::INVALID_TAG);
    }
}

Bool MediumTracker::true_hit(Expr<uint> priority) const noexcept {
    return priority <= _priority_list[0u];
}

void MediumTracker::enter(Expr<uint> priority, Expr<MediumInfo> value) noexcept {
    $if(_size == capacity) {
        device_log(
            "[error] Medium stack overflow when trying to enter priority={}, medium_tag={}",
            priority, value.medium_tag);
    }
    $else {
        _size += 1u;
        auto x = def(priority);
        auto v = def(value);
        for (auto i = 0u; i < capacity; i++) {
            auto p = _priority_list[i];
            auto m = _medium_list[i];
            auto should_swap = p > x;
            _priority_list[i] = ite(should_swap, x, p);
            _medium_list[i] = ite(should_swap, v, m);
            x = ite(should_swap, p, x);
            v = ite(should_swap, m, v);
        }
    };
}

void MediumTracker::exit(Expr<uint> priority, Expr<MediumInfo> value) noexcept {
    auto remove_num = def(0u);
    for (auto i = 0u; i < capacity - 1u; i++) {
        auto p = _priority_list[i];
        auto should_remove = (p == priority) & equal(_medium_list[i], value) & (remove_num == 0u);
        remove_num += ite(should_remove, 1u, 0u);
        _priority_list[i] = _priority_list[i + remove_num];
        _medium_list[i] = _medium_list[i + remove_num];
    }
    $if(remove_num != 0u) {
        _size -= 1u;
        _priority_list[_size] = Medium::VACUUM_PRIORITY;
        _medium_list[_size] = make_medium_info(Medium::VACUUM_PRIORITY, Medium::INVALID_TAG);
    }
    $else {
        device_log(
            "[error] Medium stack trying to exit nonexistent priority={}, medium_tag={}",
            priority, value.medium_tag);
    };
}

Bool MediumTracker::exist(Expr<uint> priority, Expr<MediumInfo> value) noexcept {
    auto exist = def(false);
    for (auto i = 0u; i < capacity - 1u; i++) {
        auto p = _priority_list[i];
        exist |= (p == priority) & equal(_medium_list[i], value);
    }
    return exist;
}

Var<MediumInfo> MediumTracker::current() const noexcept {
    auto ans = _medium_list[0];
    $if(vacuum()) {
        ans = make_medium_info(Medium::VACUUM_PRIORITY, Medium::INVALID_TAG);
    };
    return ans;
}

Bool MediumTracker::vacuum() const noexcept {
    return _priority_list[0] == Medium::VACUUM_PRIORITY;
}

}// namespace luisa::render
