//
// Created by Mike Smith on 2020/2/4.
//

#include "string_manipulation.h"
#include "exception.h"

namespace luisa { inline namespace utility {

std::string text_file_contents(const std::filesystem::path &file_path) {
    std::ifstream file{file_path};
    LUISA_ERROR_IF_NOT(file.is_open(), "failed to open file: ", file_path);
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

}}