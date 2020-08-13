//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

#include <cstdint>

namespace luisa::compute {

enum struct StorageMode : uint32_t {
    PRIVATE,  // Device-only
    MANAGED,  // Host-accessible, but explicit synchronization required
    SHARED    // Host-accessible, automatically synchronized
};

}
