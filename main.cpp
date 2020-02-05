#include <core/camera.h>
#include <core/parser.h>

using namespace luisa;

int main(int argc, char *argv[]) {

#ifndef NDEBUG
    Camera::debug();
#endif
    
    LUISA_ERROR_IF(argc != 2, "no input file");
    
    auto device = Device::create("Metal");
    Parser{device.get()}.parse(argv[1]);
    
}
