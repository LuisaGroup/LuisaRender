//
// Created by Mike Smith on 2020/9/12.
//

#define TINYEXR_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tinyexr/tinyexr.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include "texture.h"

namespace luisa::compute {

void Texture::save(Dispatcher &dispatch, const std::filesystem::path &path) const {

}

}
