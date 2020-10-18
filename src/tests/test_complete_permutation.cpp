//
// Created by Mike Smith on 2020/10/19.
//

#include <thread>
#include <compute/dsl_syntax.h>

namespace detail {

template<uint32_t N>
struct FactorialImpl {
    static constexpr auto value = N * FactorialImpl<N - 1u>::value;
};

template<>
struct FactorialImpl<0u> {
    static constexpr auto value = 1u;
};

}

template<uint32_t N>
constexpr auto factorial = detail::FactorialImpl<N>::value;

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

int main() {
    
    static constexpr std::array<uint32_t, 12u> factorial_table{
        factorial<0>,
        factorial<1>,
        factorial<2>,
        factorial<3>,
        factorial<4>,
        factorial<5>,
        factorial<6>,
        factorial<7>,
        factorial<8>,
        factorial<9>,
        factorial<10>,
        factorial<11>};
    
    using Permutation = std::array<uint8_t, 12>;
    static constexpr auto count = factorial<12u>;
    
    std::vector<Permutation> perms;
    perms.resize(count);

//    auto thread_count = std::thread::hardware_concurrency();
    auto thread_count = 8u;
    std::vector<std::thread> workers(thread_count);
    
    std::cout << "Using my algorithm..." << std::endl;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto tid = 0u; tid < thread_count; tid++) {
        
        workers[tid] = std::thread{[tid, thread_count, &perms, t0] {
            
            constexpr auto block_size = 128u;
            for (auto ii = tid * block_size; ii < count; ii += thread_count * block_size) {
                for (auto i = ii; i < std::min(ii + block_size, count); i++) {
                    
                    auto &&p = perms[i];
                    uint32_t used = 0u;
                    auto index = i;

#pragma unroll
                    for (auto j = 11; j >= 0; j--) {
                        auto factorial_j = factorial_table[j];
                        auto right_smaller_count = index / factorial_j;
                        index %= factorial_j;
                        auto empty_count = 0u;
#pragma unroll
                        for (auto slot = 0u; slot < 12u; slot++) {
                            auto mask = 1u << slot;
                            if ((used & mask) == 0u && empty_count++ == right_smaller_count) {
                                p[11 - j] = slot + 1u;
                                used |= mask;
                                break;
                            }
                        }
                    }
                }
            }
            
            auto t1 = std::chrono::high_resolution_clock::now();
            
            using namespace std::chrono_literals;
            std::cout << serialize("Thread #", tid, ": ", (t1 - t0) / 1ns * 1e-9, "s\n");
        }};
    }
    for (auto &&worker : workers) { worker.join(); }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    using namespace std::chrono_literals;
    std::cout << (t1 - t0) / 1ns * 1e-9 << "s" << std::endl;
    
    for (auto i = 0u; i < 10u; i++) {
        for (auto x : perms[i]) {
            std::cout << static_cast<uint32_t>(x) << " ";
        }
        std::cout << "\n";
    }
    
    std::ofstream file{"test.bin", std::ios::binary};
    file.write(reinterpret_cast<const char *>(perms.data()), perms.size() * sizeof(Permutation));
    
    std::cout << "Using std::next_permutation..." << std::endl;
    Permutation p{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    for (auto i = 0u; i < 10u; i++) {
        for (auto x : p) {
            std::cout << static_cast<uint32_t>(x) << " ";
        }
        std::cout << "\n";
        std::next_permutation(p.begin(), p.end());
    }
}
