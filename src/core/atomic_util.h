//
// Created by Mike Smith on 2020/9/9.
//

#pragma once

#include <atomic>

namespace luisa {

inline namespace atomic {

template<typename T>
inline void atomic_store(std::atomic<T> &object, T desired) noexcept {
    std::atomic_store_explicit(&object, desired, std::memory_order_relaxed);
}

template<typename T>
inline auto atomic_load(std::atomic<T> &obj) noexcept {
    return std::atomic_load_explicit(&obj, std::memory_order_relaxed);
}

template<typename T>
inline auto atomic_exchange(std::atomic<T> &object, T desired) noexcept {
    return std::atomic_exchange_explicit(&object, desired, std::memory_order_relaxed);
}

template<typename T>
inline auto atomic_compare_exchange_weak(std::atomic<T> &obj, T exp, T d) noexcept {
    return std::atomic_compare_exchange_weak_explicit(&obj, exp, d, std::memory_order_relaxed);
}

template<typename T>
inline auto atomic_compare_exchange_strong(std::atomic<T> &obj, T exp, T d) noexcept {
    return std::atomic_compare_exchange_strong_explicit(&obj, exp, d, std::memory_order_relaxed);
}

template<typename T>
inline auto atomic_fetch_add(std::atomic<T> &obj, T v) noexcept {
    return std::atomic_fetch_add_explicit(&obj, v, std::memory_order_relaxed);
}

template<typename T>
inline auto atomic_fetch_sub(std::atomic<T> &obj, T v) noexcept {
    return std::atomic_fetch_sub_explicit(&obj, v, std::memory_order_relaxed);
}

template<typename T>
inline auto atomic_fetch_and(std::atomic<T> &obj, T v) noexcept {
    return std::atomic_fetch_and_explicit(&obj, v, std::memory_order_relaxed);
}

template<typename T>
inline auto atomic_fetch_or(std::atomic<T> &obj, T v) noexcept {
    return std::atomic_fetch_or_explicit(&obj, v, std::memory_order_relaxed);
}

template<typename T>
inline auto atomic_fetch_xor(std::atomic<T> &obj, T v) noexcept {
    return std::atomic_fetch_xor_explicit(&obj, v, std::memory_order_relaxed);
}

}}
