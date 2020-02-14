#include <core/render.h>


using namespace luisa;

int main(int argc, char *argv[]) {
    
    LUISA_ERROR_IF(argc != 2, "no input file");
    
    auto device = Device::create("Metal");
    Parser{device.get()}.parse(argv[1])->execute();
    
}
