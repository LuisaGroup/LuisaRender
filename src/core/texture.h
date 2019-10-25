//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <util/noncopyable.h>

enum struct TextureFormat {
    RGBA32F,
    RGBA8U,
    RGB32F,
    RGB8U,
    GRAYSCALE32F,
    GRAYSCALE8U
};

class Texture : Noncopyable {

public:
    virtual ~Texture() noexcept = default;
    
};
