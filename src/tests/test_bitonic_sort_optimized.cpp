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

    constexpr auto buffer_size = 1024u;

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

    constexpr auto tg_size = 256u;

    auto small_step_kernel = device->compile_kernel([&] {
        Arg<float *> data{buffer};
        Arg<uint> cmp_stride_in{&stride};
        Arg<uint> cmp_step_in{&step};

        Auto tgid = thread_id() % tg_size;
        Threadgroup<std::array<float, 256>> cache;
        cache[tgid] = data[thread_id()];
        threadgroup_barrier();

        Auto cmp_step = cmp_step_in;
        While(cmp_step >= 2u) {

            Auto half_cmp_step = cmp_step / 2u;

            Auto lhs_index = thread_id() / half_cmp_step * cmp_step + thread_id() % half_cmp_step;
            Auto rhs_index = lhs_index + half_cmp_step;

            Auto lhs = cache[lhs_index % tg_size];
            Auto rhs = cache[rhs_index % tg_size];

            Auto cmp_stride = cmp_stride_in;
            Auto reverse_ordered = lhs_index / cmp_stride % 2u;

            Auto smaller = min(lhs, rhs);
            Auto greater = max(lhs, rhs);
            Auto ascending_pair = make_uint2(smaller, greater);
            Auto descending_pair = make_uint2(greater, smaller);

            Auto result = select(reverse_ordered == 1u, descending_pair, ascending_pair);
            cache[lhs_index % tg_size] = result.x();
            cache[rhs_index % tg_size] = result.y();

            cmp_step /= 2u;

            threadgroup_barrier();
        };

        data[thread_id()] = cache[tgid];
    });

    auto small_stride_kernel = device->compile_kernel([&] {
        
        Arg<float *> data{buffer};
        
        Threadgroup<std::array<float, 256u>> cache;
        cache[thread_id() % tg_size] = data[thread_id()];
        threadgroup_barrier();

        Auto cmp_stride = tg_size;
        While(cmp_stride <= buffer_size) {
            Auto cmp_step = cmp_stride;
            While(cmp_step >= 2u) {
                Auto half_cmp_step = cmp_step / 2u;
                Auto tid = thread_id();
                Auto lhs_index = tid / half_cmp_step * cmp_step + tid % half_cmp_step;
                Auto rhs_index = lhs_index + half_cmp_step;

                Auto lhs = cache[lhs_index % tg_size];
                Auto rhs = cache[rhs_index % tg_size];

                Auto reverse_ordered = lhs_index / cmp_stride % 2u;

                Auto smaller = min(lhs, rhs);
                Auto greater = max(lhs, rhs);
                Auto ascending_pair = make_uint2(smaller, greater);
                Auto descending_pair = make_uint2(greater, smaller);

                Auto result = select(reverse_ordered == 1u, descending_pair, ascending_pair);
                cache[lhs_index % tg_size] = result.x();
                cache[rhs_index % tg_size] = result.y();

                cmp_step /= 2;

                threadgroup_barrier();
            };
            cmp_stride *= 2u;
        };
        data[thread_id()] = cache[thread_id() % tg_size];
    });

    std::default_random_engine random{std::random_device{}()};

    for (auto i = 0u; i < 20u; i++) {
        std::shuffle(host_buffer.begin(), host_buffer.end(), random);
        device->launch(buffer.copy_from(host_buffer.data()), [i] { LUISA_INFO("Copied #", i); });
        device->launch([&](Dispatcher &dispatch) {
          for (stride = 2u; stride < tg_size; stride *= 2u) {
              for (step = stride; step >= 2u; step /= 2) {
                  dispatch(kernel->parallelize(buffer_size / 2u, tg_size));
              }
              // step starts from tg_size / 2
//              dispatch(small_step_kernel->parallelize(buffer_size / 2u, tg_size));
          }
            dispatch(small_stride_kernel->parallelize(buffer_size / 2u, tg_size)); },
                       [i] { LUISA_INFO("Sorted #", i); });
    }
    device->launch(buffer.copy_to(host_buffer.data()));
    device->synchronize();

    LUISA_INFO("Checking...");
    std::ostringstream ss;
    for (auto v : host_buffer) { ss << v << " "; }
    LUISA_INFO(ss.str());
    LUISA_ERROR_IF_NOT(std::is_sorted(host_buffer.cbegin(), host_buffer.cend()), "Fuck!");
    LUISA_INFO("Good!");
}
