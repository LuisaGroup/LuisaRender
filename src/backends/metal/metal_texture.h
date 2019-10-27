//
// Created by Mike Smith on 2019/10/27.
//

#pragma once

#ifndef __OBJC__
#error "This file should only be used in Objective-C/C++ sources."
#endif

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <core/texture.h>

class MetalTexture : public Texture {

private:
    id<MTLTexture> _texture;

public:
    MetalTexture(id<MTLTexture> texture, uint2 size, TextureFormatTag format_tag, TextureAccessTag access_tag)
        : _texture{texture}, Texture{size, format_tag, access_tag} {}
    
    void copy_from_buffer(struct KernelDispatcher &dispatch, Buffer &buffer) override;
    void copy_to_buffer(struct KernelDispatcher &dispatch, Buffer &buffer) override;
    
    [[nodiscard]] id<MTLTexture> texture() const noexcept { return _texture; }
    
};


