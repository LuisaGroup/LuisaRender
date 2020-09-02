//
// Created by Mike Smith on 2020/8/8.
//

#include "logging.h"

namespace luisa::logging {

spdlog::logger &logger() noexcept {
    static auto l = spdlog::stdout_color_mt("console");
    return *l;
}

}
