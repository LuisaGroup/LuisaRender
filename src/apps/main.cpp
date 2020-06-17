#include <core/render.h>

using namespace luisa;

int main(int argc, char *argv[]) {

    LUISA_INFO("Welcome to LuisaRender!");

    try {
        LUISA_EXCEPTION_IF(argc < 2, "No input file specified");
        auto runtime_directory = std::filesystem::canonical(argv[0]).parent_path().parent_path();
        for (auto i = 1; i < argc; i++) {
            auto scene_path = std::filesystem::canonical(argv[i]);
            auto working_directory = scene_path.parent_path();
            Context context{runtime_directory, working_directory};
            Parser{Device::create(&context, "metal").get()}.parse(scene_path)->execute();
        }
    } catch (const std::exception &e) {
        LUISA_ERROR(e.what());
    }
}
