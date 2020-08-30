//
// Created by Mike on 8/27/2020.
//

#include <compute/device.h>
#include <compute/dsl.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

int main(int argc, char *argv[]) {

    Context context{argc, argv};
    auto device = Device::create(&context, "cuda");
    auto buffer = device->allocate_buffer<float>(1024);
    auto kernel = device->compile_kernel([&] {
        LUISA_INFO(0);
        Arg<float *> a{buffer};
        LUISA_INFO(1);
        auto tid = thread_id();
        If(tid < 1024) {
            a[tid] = cos(a[tid]);
        };
    });
}
