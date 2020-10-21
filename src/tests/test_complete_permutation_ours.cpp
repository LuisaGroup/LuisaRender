//
// Created by Mike Smith on 2020/10/19.
//

#include <thread>
#include <vector>
#include <chrono>
#include <iostream>
#include <array>
#include <sstream>
#include <algorithm>

static constexpr auto digit_count = 12u;

template<int len>
struct Permutation {
    
    std::array<uint8_t, len> data;
    
    Permutation() {
        for (int i = 0; i < len; i++) {
            data[i] = i + 1;
        }
    }
    
    void get_next_permutation_dic_ord() {
#pragma unroll
        for (int i = len - 1; i >= 1; i--)//从后往前遍历
        {
            if (data[i] > data[i - 1])//一旦遇到下降的地方
            {
                int min = 65536;//记录下此后最小值的位置和值
                int min_index = -1;
                for (int j = i; j <= len - 1; j++) {
                    if (data[j] < min && data[j] > data[i - 1]) {
                        min = data[j];
                        min_index = j;
                    }
                }
                std::swap(data[i - 1], data[min_index]);//将下降后的位置和之后的最小值互换
                std::reverse(data.begin() + i, data.begin() + len);//然后将后面部分排序
                break;
            }
        }
    }
};

constexpr auto factorial(uint32_t n) noexcept {
    auto x = 1u;
    for (auto i = 1u; i <= n; i++) { x *= i; }
    return x;
}

int main(int argc, char *argv[]) {
    
    std::array<uint32_t, digit_count> factorial_table{};
    for (auto i = 0u; i < digit_count; i++) { factorial_table[i] = factorial(i); }
    
    auto count = factorial(digit_count);
    
    std::vector<Permutation<digit_count>> perms(count);
    
    std::istringstream is{argv[1]};
    auto worker_count = 8u;
    is >> worker_count;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    
    std::cout << "Using our algorithm..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto tid = 0u; tid < worker_count; tid++) {
        
        workers.emplace_back([tid, worker_count, &perms, t0, count, factorial_table] {
            
            Permutation<digit_count> p;
            
            constexpr auto block_size = 1024u;
            for (auto i = tid * block_size; i < count; i += worker_count * block_size) {
                
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
                        p.get_next_permutation_dic_ord();
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
    Permutation<digit_count> p;
    for (auto i = 0u; i < count; i++) {
        if (perms[i].data != p.data) { exit(-1); }
        std::next_permutation(p.data.begin(), p.data.end());
    }
    std::cout << "Pass!" << std::endl;
    
    std::cout << "Using serial..." << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    for (auto i = 0u; i < count; i++) {
        perms[i] = p;
        p.get_next_permutation_dic_ord();
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0) / 1ns * 1e-9 << "s" << std::endl;
}
