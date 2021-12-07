//
// Created by Mike on 2021/12/7.
//

#include <runtime/device.h>
#include <runtime/context.h>

int main(int argc, char *argv[]) {
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device("cuda");
}
