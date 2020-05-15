#include <core/render.h>

using namespace luisa;

int main(int argc, char *argv[]) {
    
    LUISA_INFO("Welcome!");
    
    auto runtime_directory = std::filesystem::canonical(argv[0]).parent_path().parent_path();
    auto working_directory = std::filesystem::canonical(argv[1]).parent_path();
    Context context{runtime_directory, working_directory};
    
    try {
        LUISA_EXCEPTION_IF(argc != 2, "No input file specified");
        Parser{Device::create(&context, "metal").get()}.parse(argv[1])->execute();
    } catch (const std::exception &e) {
        LUISA_ERROR(e.what());
    }
}
