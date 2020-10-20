//
// Created by Mike Smith on 2020/10/19.
//

#include <compute/device.h>
#include <compute/pipeline.h>
#include <compute/dsl_syntax.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

constexpr auto factorial(uint32_t n) noexcept {
    auto x = 1u;
    for (auto i = 1u; i <= n; i++) { x *= i; }
    return x;
}

static constexpr auto digit_count = 12u;

int main(int argc, char *argv[]) {
    
    constexpr std::array<uint32_t, 12u> factorial_table{
        factorial(0u), factorial(1u), factorial(2u), factorial(3u),
        factorial(4u), factorial(5u), factorial(6u), factorial(7u),
        factorial(8u), factorial(9u), factorial(10u), factorial(11u)};
    
    Context context{argc, argv};
    
    using Permutation = std::array<uchar, digit_count>;
    
    constexpr auto count = factorial(digit_count);
    std::vector<Permutation> perms(count);
    
    auto device = Device::create(&context);
    
    constexpr auto part_size = (1u << 24u);
    constexpr auto thread_count = 1u << 20u;
    constexpr auto block_size = (part_size + thread_count - 1u) / thread_count;
    constexpr auto perm_buffer_size = digit_count * part_size;  // process at most 2^28 = 256M elements at once
    auto perm_buffer = device->allocate_buffer<Permutation>(perm_buffer_size);
    
    auto u_start = 0u;
    auto kernel = device->compile_kernel("complete_permutation_part", [&] {
        
        Var start = uniform(&u_start);
        Var<Permutation> p;
        
        for (auto i = 0; i < part_size; i += thread_count) {
            
            Var used = 0u;
            Var index = start + i;
            for (auto j = 0u; j < digit_count; j++) {
                auto factorial_j = factorial_table[digit_count - 1u - j];
                Var right_smaller_count = index / factorial_j;
                index %= factorial_j;
                Var empty_count = 0u;
                Var slot = 0u;
                While (slot < digit_count) {
                    auto mask = 1u << slot;
                    If ((used & mask) == 0u && empty_count + 1 == right_smaller_count) {
                        p[j] = slot + 1u;
                        used |= mask;
                        Break;
                    };
                    empty_count += 1u;
                    slot += 1u;
                };
                perm_buffer[i] = p;
            }
        }
    });
    
    kernel.wait_for_compilation();
    LUISA_INFO("Compilation finished!");
    
    Pipeline pipeline{device.get()};
    for (auto start = 0u; start < count; start += part_size) {
        u_start = start;
        pipeline << kernel.parallelize(thread_count)
                 << perm_buffer.copy_to(perms.data() + start);
    }
    pipeline << synchronize();
}
