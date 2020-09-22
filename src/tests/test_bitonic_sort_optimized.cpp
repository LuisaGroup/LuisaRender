//
// Created by Mike on 9/3/2020.
//

#include <algorithm>
#include <random>
#include <vector>

#include <compute/device.h>
#include <compute/dsl_syntax.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

int main(int argc, char *argv[]) {
    
    Context context{argc, argv};
    
    auto device = Device::create(&context);
    
    static constexpr auto buffer_size = 1024u * 1024u;
    static constexpr auto block_size = 1024u;
    
    std::vector<float> host_buffer(buffer_size);
    for (auto i = 0u; i < buffer_size; i++) { host_buffer[i] = static_cast<float>(i); }
    
    auto buffer = device->allocate_buffer<float>(buffer_size);
    
    auto stride = 0u;
    auto step = 0u;
    
    auto kernel = device->compile_kernel("basic", [&] {
        
        auto cmp_stride_in = uniform(&stride);
        auto cmp_step_in = uniform(&step);
        
        Var cmp_step = cmp_step_in;
        Var half_cmp_step = cmp_step / 2u;
        Var tid = thread_id();
        Var lhs_index = tid / half_cmp_step * cmp_step + tid % half_cmp_step;
        Var rhs_index = lhs_index + half_cmp_step;
        
        Var lhs = buffer[lhs_index];
        Var rhs = buffer[rhs_index];
        
        Var predicate = cast<bool>(lhs_index & cmp_stride_in) ^ (lhs < rhs);
        buffer[lhs_index] = select(predicate, lhs, rhs);
        buffer[rhs_index] = select(predicate, rhs, lhs);
    });
    
    auto small_stride_kernel = device->compile_kernel("small_stride", [&] {
        
        Threadgroup<float> cache{block_size};
        
        constexpr auto half_block_size = block_size / 2u;
        Var tgid = thread_id() % half_block_size;
        cache[tgid * 2u] = buffer[thread_id() * 2u];
        cache[tgid * 2u + 1u] = buffer[thread_id() * 2u + 1u];
        threadgroup_barrier();
        
        auto tid = thread_id();
        for (auto cmp_stride = 2u; cmp_stride <= block_size; cmp_stride *= 2u) {
            for (auto cmp_step = cmp_stride; cmp_step >= 2u; cmp_step /= 2u) {
                
                auto half_cmp_step = cmp_step / 2u;
                
                Var lhs_index = tid / half_cmp_step * cmp_step + tid % half_cmp_step;
                Var rhs_index = lhs_index + half_cmp_step;
                
                Var lhs = cache[lhs_index % block_size];
                Var rhs = cache[rhs_index % block_size];
    
                Var predicate = cast<bool>(lhs_index & cmp_stride) ^ (lhs < rhs);
                cache[lhs_index % block_size] = select(predicate, lhs, rhs);
                cache[rhs_index % block_size] = select(predicate, rhs, lhs);
                
                threadgroup_barrier();
            }
        }
        buffer[thread_id() * 2u] = cache[tgid * 2u];
        buffer[thread_id() * 2u + 1u] = cache[tgid * 2u + 1u];
    });
    
    auto small_step_kernel = device->compile_kernel("small_step", [&] {
        
        auto cmp_stride_in = uniform(&stride);
        
        Threadgroup<float> cache{block_size};
        constexpr auto half_block_size = block_size / 2u;
        Var tgid = thread_id() % half_block_size;
        cache[tgid * 2u] = buffer[thread_id() * 2u];
        cache[tgid * 2u + 1u] = buffer[thread_id() * 2u + 1u];
        threadgroup_barrier();
        
        auto tid = thread_id();
        auto cmp_stride = cmp_stride_in;
        
        for (auto cmp_step = block_size; cmp_step >= 2u; cmp_step /= 2) {
            
            auto half_cmp_step = cmp_step / 2u;
            
            Var lhs_index = tid / half_cmp_step * cmp_step + tid % half_cmp_step;
            Var rhs_index = lhs_index + half_cmp_step;
            
            Var lhs = cache[lhs_index % block_size];
            Var rhs = cache[rhs_index % block_size];
            
            Var predicate = cast<bool>(lhs_index & cmp_stride) ^ (lhs < rhs);
            cache[lhs_index % block_size] = select(predicate, lhs, rhs);
            cache[rhs_index % block_size] = select(predicate, rhs, lhs);
            
            threadgroup_barrier();
        }
        buffer[thread_id() * 2u] = cache[tgid * 2u];
        buffer[thread_id() * 2u + 1u] = cache[tgid * 2u + 1u];
    });
    
    std::default_random_engine random{std::random_device{}()};
    
    for (auto i = 0u; i < 20u; i++) {
        std::shuffle(host_buffer.begin(), host_buffer.end(), random);
        device->launch(buffer.copy_from(host_buffer.data()), [i] { LUISA_INFO("Copied #", i); });
        device->launch([&](Dispatcher &dispatch) {
            dispatch(small_stride_kernel.parallelize(buffer_size / 2u, block_size / 2u));
            for (stride = block_size * 2u; stride <= buffer_size; stride *= 2u) {
                for (step = stride; step >= block_size * 2u; step /= 2) {
                    dispatch(kernel.parallelize(buffer_size / 2u, block_size / 2u));
                }
                dispatch(small_step_kernel.parallelize(buffer_size / 2u, block_size / 2u));
            }
        }, [i] { LUISA_INFO("Sorted #", i); });
    }
    device->launch(buffer.copy_to(host_buffer.data()));
    device->synchronize();
    
    LUISA_INFO("Checking...");
    for (auto i = 0u; i < buffer_size; i++) {
        LUISA_ERROR_IF_NOT(host_buffer[i] == static_cast<float>(i), "Fuck!");
    }
    LUISA_INFO("Good!");
}
