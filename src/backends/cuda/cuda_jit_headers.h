//
// Created by Mike on 8/28/2020.
//

#pragma once

#include <core/context.h>
#include <map>

namespace luisa::cuda {

inline const auto &get_jit_headers(Context *context) noexcept {
    static std::map<const char *, std::string> headers{
        {"scalar_types.h", text_file_contents(context->runtime_path("include") / "core" / "scalar_types.h")},
        {"vector_types.h", text_file_contents(context->runtime_path("include") / "core" / "vector_types.h")},
        {"matrix_types.h", text_file_contents(context->runtime_path("include") / "core" / "matrix_types.h")},
        {"data_types.h", text_file_contents(context->runtime_path("include") / "core" / "data_types.h")},
        {"mathematics.h", text_file_contents(context->runtime_path("include") / "core" / "mathematics.h")}};
    return headers;
}

}// namespace luisa::cuda
