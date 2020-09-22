//
// Created by Mike Smith on 2020/9/14.
//

#include <render/task.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

int main(int argc, char *argv[]) {
    
    try {
        Context context{argc, argv};
        auto device = Device::create(&context);
        Parser parser{device.get()};
        parser.parse(context.cli_positional_option())->execute();
    } catch (const std::exception &e) {
        LUISA_ERROR("Error occurred: ", e.what());
    }
}
