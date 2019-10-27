//
// Created by Mike Smith on 2019/10/24.
//

#pragma once

#include <compatibility.h>
#include <util/noncopyable.h>

#include "buffer.h"

enum struct TextureFormatTag {
    RGBA32F,
    GRAYSCALE32F
};

enum struct TextureAccessTag {
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE
};

class Texture : Noncopyable {

private:
    uint2 _size;
    TextureFormatTag _format;
    TextureAccessTag _access;

public:
    Texture(uint2 size, TextureFormatTag format_tag, TextureAccessTag access_tag) noexcept
        : _size{size}, _format{format_tag}, _access{access_tag} {}
    virtual ~Texture() noexcept = default;
    virtual void copy_from_buffer(struct KernelDispatcher &dispatch, Buffer &buffer) = 0;
    virtual void copy_to_buffer(struct KernelDispatcher &dispatch, Buffer &buffer) = 0;
};
