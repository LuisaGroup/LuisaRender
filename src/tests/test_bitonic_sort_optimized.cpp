//
// Created by Mike on 9/3/2020.
//

#include <algorithm>
#include <random>
#include <vector>

#include <compute/device.h>
#include <compute/dsl.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

int main(int argc, char *argv[]) {
    
    Context context{argc, argv};
    
    auto device = Device::create(&context);
    
    constexpr auto buffer_size = 1024u * 1024u;
    
    std::vector<float> host_buffer(buffer_size);
    for (auto i = 0u; i < buffer_size; i++) { host_buffer[i] = static_cast<float>(i); }
    
    auto buffer = device->allocate_buffer<float>(buffer_size);
    
    auto stride = 0u;
    auto step = 0u;
    
    auto kernel = device->compile_kernel([&] {
        Arg<float *> data{buffer};
        Arg<uint> cmp_stride_in{&stride};
        Arg<uint> cmp_step_in{&step};
        
        Auto cmp_step = cmp_step_in;
        Auto half_cmp_step = cmp_step / 2u;
        Auto tid = thread_id();
        Auto lhs_index = tid / half_cmp_step * cmp_step + tid % half_cmp_step;
        Auto rhs_index = lhs_index + half_cmp_step;
        
        Auto lhs = data[lhs_index];
        Auto rhs = data[rhs_index];
        
        Auto cmp_stride = cmp_stride_in;
        Auto reverse_ordered = lhs_index / cmp_stride % 2u;
        
        Auto smaller = min(lhs, rhs);
        Auto greater = max(lhs, rhs);
        Auto ascending_pair = make_uint2(smaller, greater);
        Auto descending_pair = make_uint2(greater, smaller);
        
        Auto result = select(reverse_ordered == 1u, descending_pair, ascending_pair);
        data[lhs_index] = result.x();
        data[rhs_index] = result.y();
    });
    
    constexpr auto block_size = 1024u;
    
    auto small_step_kernel = device->compile_kernel([&] {
        Arg<float *> data{buffer};
        Arg<uint> cmp_stride_in{&stride};
        Arg<uint> cmp_step_in{&step};
        
        Auto tgid = thread_id() % block_size;
        Threadgroup<std::array<float, 256>> cache;
        cache[tgid] = data[thread_id()];
        threadgroup_barrier();
        
        Auto cmp_step = cmp_step_in;
        While(cmp_step >= 2u) {
            
            Auto half_cmp_step = cmp_step / 2u;
            
            Auto lhs_index = thread_id() / half_cmp_step * cmp_step + thread_id() % half_cmp_step;
            Auto rhs_index = lhs_index + half_cmp_step;
            
            Auto lhs = cache[lhs_index % block_size];
            Auto rhs = cache[rhs_index % block_size];
            
            Auto cmp_stride = cmp_stride_in;
            Auto reverse_ordered = lhs_index / cmp_stride % 2u;
            
            Auto smaller = min(lhs, rhs);
            Auto greater = max(lhs, rhs);
            Auto ascending_pair = make_uint2(smaller, greater);
            Auto descending_pair = make_uint2(greater, smaller);
            
            Auto result = select(reverse_ordered == 1u, descending_pair, ascending_pair);
            cache[lhs_index % block_size] = result.x();
            cache[rhs_index % block_size] = result.y();
            
            cmp_step /= 2u;
            
            threadgroup_barrier();
        };
        
        data[thread_id()] = cache[tgid];
    });
    
    auto small_stride_kernel = device->compile_kernel([&] {
        
        Arg<float *> data{buffer};
        
        Threadgroup<std::array<float, 1024u>> cache;
        cache[thread_x() * 2u] = data[thread_y() * block_size + thread_x() * 2u];
        cache[thread_x() * 2u + 1u] = data[thread_y() * block_size + thread_x() * 2u + 1u];
        threadgroup_barrier();
        
        for (auto cmp_stride = 2u; cmp_stride <= block_size; cmp_stride *= 2u) {
            for (auto cmp_step = cmp_stride; cmp_step >= 2u; cmp_step /= 2u) {
                
                auto half_cmp_step = cmp_step / 2u;
                Auto lhs_index = thread_x() / half_cmp_step * cmp_step + thread_x() % half_cmp_step;
                Auto rhs_index = lhs_index + half_cmp_step;
                
                Auto lhs = cache[lhs_index];
                Auto rhs = cache[rhs_index];
                
                Auto reverse_ordered = lhs_index / cmp_stride % 2u;
                
                Auto smaller = min(lhs, rhs);
                Auto greater = max(lhs, rhs);
                Auto ascending_pair = make_uint2(smaller, greater);
                Auto descending_pair = make_uint2(greater, smaller);
                
                Auto result = select(reverse_ordered == 1u, descending_pair, ascending_pair);
                cache[lhs_index] = result.x();
                cache[rhs_index] = result.y();
                threadgroup_barrier();
            }
        }
        data[thread_y() * block_size + thread_x() * 2u] = cache[thread_x() * 2u];
        data[thread_y() * block_size + thread_x() * 2u + 1u] = cache[thread_x() * 2u + 1u];
    });
    
    std::default_random_engine random{std::random_device{}()};
    
    for (auto i = 0u; i < 20u; i++) {
        std::shuffle(host_buffer.begin(), host_buffer.end(), random);
        device->launch(buffer.copy_from(host_buffer.data()), [i] { LUISA_INFO("Copied #", i); });
        device->launch([&](Dispatcher &dispatch) {
            dispatch(small_stride_kernel->parallelize(make_uint2(block_size / 2u, buffer_size / block_size), make_uint2(block_size / 2u, 1u)));
//            for (stride = 2u; stride <= buffer_size; stride *= 2u) {
//                for (step = stride; step >= 2u; step /= 2) {
//                    dispatch(kernel->parallelize(buffer_size / 2u, block_size));
//                }
//            }
        }, [i] { LUISA_INFO("Sorted #", i); });
    }
    device->launch(buffer.copy_to(host_buffer.data()));
    device->synchronize();
    
    LUISA_INFO("Checking...");
//    for (auto i = 0u; i + block_size <= buffer_size; i += block_size) {
//        std::ostringstream ss;
//        for (auto j = i; j < i + block_size; j++) {
//            ss << host_buffer[j] << " ";
//        }
//        LUISA_INFO(ss.str());
//    }
    
    for (auto i = 0u; i + block_size <= buffer_size; i += block_size) {
        LUISA_ERROR_IF_NOT(std::is_sorted(host_buffer.cbegin() + i, host_buffer.cbegin() + i + block_size), "Fuck!");
    }
    LUISA_INFO("Good!");
}
