//
// Created by Mike on 8/27/2020.
//

#include <compute/device.h>
#include <compute/dsl.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::dsl;

int main(int argc, char *argv[]) {

    try {
        Context context{argc, argv};
        auto device = Device::create(&context);
        auto buffer = device->allocate_buffer<float>(1024);

        auto kernel = device->compile_kernel([&] {
            Arg<float *> a{buffer};
            auto tid = thread_id();
            If(tid < 1024) {
                a[tid] = cos(select(a[tid] < 0, -a[tid], a[tid]));
            };
        });
    } catch (const std::exception &e) {
        LUISA_ERROR("Error occurred: ", e.what());
    }
}
