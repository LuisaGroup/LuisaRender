//
// Created by Mike Smith on 2020/10/19.
//

#include <thread>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <array>
#include <sstream>

static constexpr auto digit_count = 12u;

constexpr auto factorial(uint32_t n) noexcept {
    auto x = 1u;
    for (auto i = 1u; i <= n; i++) { x *= i; }
    return x;
}

struct alignas((digit_count + 7u) / 8u * 8u) Permutation {
    std::array<uint8_t, digit_count> data{};
    constexpr Permutation() noexcept {
        for (auto i = 0u; i < digit_count; i++) {
            data[i] = i + 1u;
        }
    }
};

int main() {
    
    std::array<uint32_t, digit_count> factorial_table{};
    for (auto i = 0u; i < digit_count; i++) { factorial_table[i] = factorial(i); }
    
    auto count = factorial(digit_count);
    
    std::vector<Permutation> perms(count);
    
    auto worker_count = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    
    std::cout << "Using our algorithm..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto tid = 0u; tid < worker_count; tid++) {
        
        workers.emplace_back([tid, worker_count, &perms, t0, count, factorial_table] {
            
            constexpr auto block_size = 1024u;
            for (auto i = tid * block_size; i < count; i += worker_count * block_size) {
                
                Permutation p;
                uint32_t used = 0u;
                auto index = i;

#pragma unroll
                for (auto j = 0u; j < digit_count; j++) {
                    auto factorial_j = factorial_table[digit_count - 1u - j];
                    auto right_smaller_count = index / factorial_j;
                    index %= factorial_j;
                    auto empty_count = 0u;

#pragma unroll
                    for (auto slot = 0u; slot < digit_count; slot++) {
                        auto mask = 1u << slot;
                        if ((used & mask) == 0u && empty_count++ == right_smaller_count) {
                            p.data[j] = slot + 1u;
                            used |= mask;
                            break;
                        }
                    }
                }
                
                for (auto j = 0; j < block_size; j++) {
                    if (auto jj = i + j; jj < count) {
                        perms[jj] = p;
                        std::next_permutation(p.data.begin(), p.data.end());
                    } else {
                        break;
                    }
                }
            }
            
            auto t1 = std::chrono::high_resolution_clock::now();
            
            using namespace std::chrono_literals;
            std::ostringstream os;
            os << "Thread #" << tid << ": " << (t1 - t0) / 1ns * 1e-9 << "s\n";
            std::cout << os.str();
        });
    }
    for (auto &&worker : workers) { worker.join(); }
    using namespace std::chrono_literals;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0) / 1ns * 1e-9 << "s" << std::endl;
    
    std::cout << "Checking... ";
    Permutation p;
    for (auto i = 0u; i < count; i++) {
        if (perms[i].data != p.data) { exit(-1); }
        std::next_permutation(p.data.begin(), p.data.end());
    }
    std::cout << "Pass!" << std::endl;
    
    std::cout << "Using std::next_permutation..." << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    for (auto i = 0u; i < count; i++) {
        perms[i] = p;
        std::next_permutation(p.data.begin(), p.data.end());
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0) / 1ns * 1e-9 << "s" << std::endl;
}
