//
// Created by Mike Smith on 2020/9/3.
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
    context.add_cli_option<uint>("b,blocksize", "Block size (results are sorted block-wise)", "1024");

    auto device = Device::create(&context);

    constexpr auto buffer_size = 1024u * 1024u;
    
    std::vector<float> host_buffer(buffer_size);
    for (auto i = 0u; i < buffer_size; i++) { host_buffer[i] = static_cast<float>(i); }

    auto buffer = device->allocate_buffer<float>(buffer_size);

    auto stride = 1u;
    auto step = 1u;

    auto block_size = context.cli_option<uint>("blocksize");

    auto kernel = device->compile_kernel([&] {
        
        auto cmp_stride_in = uniform(&stride);
        auto cmp_step_in = uniform(&step);

        Var cmp_step = cmp_step_in;
        Var half_cmp_step = cmp_step / 2u;
        Var tid_x = thread_xy().x();
        Var lhs_index = tid_x / half_cmp_step * cmp_step + tid_x % half_cmp_step;
        Var rhs_index = lhs_index + half_cmp_step;

        Var tid_y = thread_xy().y();
        Var lhs = buffer[tid_y * block_size + lhs_index];
        Var rhs = buffer[tid_y * block_size + rhs_index];

        Var cmp_stride = cmp_stride_in;
        Var reverse_ordered = lhs_index / cmp_stride % 2u;

        Var smaller = min(lhs, rhs);
        Var greater = max(lhs, rhs);
        Var ascending_pair = make_uint2(smaller, greater);
        Var descending_pair = make_uint2(greater, smaller);

        Var result = select(reverse_ordered == 1u, descending_pair, ascending_pair);
        buffer[tid_y * block_size + lhs_index] = result.x();
        buffer[tid_y * block_size + rhs_index] = result.y();
    });
    std::default_random_engine random{std::random_device{}()};
    
    for (auto i = 0u; i < 20u; i++) {
        std::shuffle(host_buffer.begin(), host_buffer.end(), random);
        device->launch(buffer.copy_from(host_buffer.data()), [i] { LUISA_INFO("Copied #", i); });
        device->launch([&](Dispatcher &dispatch) {
        for (stride = 2u; stride <= block_size; stride *= 2u) {
            for (step = stride; step >= 2; step /= 2) {
                dispatch(kernel.parallelize(make_uint2(block_size / 2u, buffer_size / block_size), make_uint2(256u, 1u)));
            }
        } }, [i] { LUISA_INFO("Sorted #", i); });
    }
    device->launch(buffer.copy_to(host_buffer.data()));
    device->synchronize();

    LUISA_INFO("Checking...");
    for (auto i = 0u; i < buffer_size; i += block_size) {
        LUISA_ERROR_IF_NOT(std::is_sorted(host_buffer.cbegin() + i, host_buffer.cbegin() + i + block_size), "Fuck!");
    }
    LUISA_INFO("Good!");
}
