#include <core/render.h>

using namespace luisa;

int main(int argc, char *argv[]) {
    LUISA_ERROR_IF(argc != 2, "no input file");
    Parser{Device::create("Metal").get()}.parse(argv[1])->execute();
}
