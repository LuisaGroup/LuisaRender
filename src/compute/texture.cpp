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

void Texture::save(Dispatcher &dispatch, const std::filesystem::path &path) {
    
    auto path_str = std::filesystem::absolute(path).string();
    auto extension = to_lower(path.extension());
    
    if (extension == ".exr") {
        LUISA_ERROR_IF_NOT(is_hdr(), "Only HDR textures are allowed to be saved as OpenEXR files.");
        LUISA_ERROR("Not implemented!");
    } else if (extension == ".hdr") {
        LUISA_ERROR_IF_NOT(format() == PixelFormat::RGBA32F, "Only RGBA32F textures are allowed to be saved as HDRI images.");
        LUISA_WARNING("Alpha channels will be discarded when textures saved as HDRI images.");
        auto pixels = std::make_shared<std::vector<float>>(pixel_count() * 4);
        copy_to(dispatch, pixels->data());
        dispatch.when_completed([pixels, count = pixel_count(), w = width(), h = height(), path_str, path] {
            stbi_write_hdr(path_str.c_str(), w, h, 4, pixels->data());
            LUISA_INFO("Done saving texture: ", path);
        });
    } else {
        LUISA_ERROR_IF_NOT(extension == ".bmp" || extension == ".png" || extension == ".jpg" || extension == ".jpeg",
                           "Failed to save texture with unsupported file extension: ", path);
        auto pixels = std::make_shared<std::vector<uchar>>(pixel_count() * channels());
        copy_to(dispatch, pixels->data());
        dispatch.when_completed([pixels, count = pixel_count(), w = width(), h = height(), c = channels(), path_str, path, extension] {
            if (extension == ".bmp") {
                LUISA_WARNING_IF(c == 2, "Saving RG8U textures to Bitmap images may lead to unexpected results.");
                stbi_write_bmp(path_str.c_str(), w, h, c, pixels->data());
            } else if (extension == ".png") {
                stbi_write_png(path_str.c_str(), w, h, c, pixels->data(), 0);
            } else {
                LUISA_WARNING_IF(c == 2, "Saving RG8U textures to JPEG images may lead to unexpected results.");
                stbi_write_jpg(path_str.c_str(), w, h, c, pixels->data(), 100);
            }
            LUISA_INFO("Done saving texture: ", path);
        });
    }
}

}
