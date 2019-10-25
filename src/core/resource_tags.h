//
// Created by Mike Smith on 2019/10/26.
//

#pragma once

enum struct StorageTag {
    DEVICE_PRIVATE, MANAGED
};

enum struct AccessTag {
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE
};
