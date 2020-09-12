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
        auto pixels = std::make_shared<std::vector<float>>(pixel_count() * channels());
        copy_to(dispatch, pixels->data());
        
        dispatch.when_completed([pixels = std::move(pixels), count = pixel_count(), w = width(), h = height(), c = channels(), path_str, path] {
            
            EXRHeader header;
            InitEXRHeader(&header);
            
            EXRImage image;
            InitEXRImage(&image);
            
            std::array<float *, 4> image_ptr{nullptr, nullptr, nullptr, nullptr};
            image.num_channels = c;
            image.width = w;
            image.height = h;
            image.images = reinterpret_cast<unsigned char **>(image_ptr.data());
            
            std::array<int, 4> pixel_types{TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT};
            std::array<EXRChannelInfo, 4> channels{};
            header.num_channels = c;
            header.channels = channels.data();
            header.pixel_types = pixel_types.data();
            header.requested_pixel_types = pixel_types.data();
            
            std::vector<float> images;
            if (c == 1) {
                image_ptr[0] = pixels->data();
                strcpy(header.channels[0].name, "A");
            } else if (c == 2) {
                images.resize(2 * count);
                image_ptr[0] = images.data();
                image_ptr[1] = image_ptr[0] + count;
                auto rg = reinterpret_cast<float2 *>(pixels->data());
                for (int i = 0u; i < count; i++) {
                    image_ptr[1][i] = rg[i].x;
                    image_ptr[0][i] = rg[i].y;
                }
                strcpy(header.channels[0].name, "G");
                strcpy(header.channels[1].name, "R");
            } else {
                images.resize(4 * count);
                image_ptr[0] = images.data();
                image_ptr[1] = image_ptr[0] + count;
                image_ptr[2] = image_ptr[1] + count;
                image_ptr[3] = image_ptr[2] + count;
                auto rgba = reinterpret_cast<float4 *>(pixels->data());
                for (int i = 0u; i < count; i++) {
                    image_ptr[3][i] = rgba[i].x;
                    image_ptr[2][i] = rgba[i].y;
                    image_ptr[1][i] = rgba[i].z;
                    image_ptr[0][i] = rgba[i].w;
                }
                strcpy(header.channels[0].name, "A");
                strcpy(header.channels[1].name, "B");
                strcpy(header.channels[2].name, "G");
                strcpy(header.channels[3].name, "R");
            }
            
            const char *err = nullptr; // or nullptr in C++11 or later.
            auto err_guard = guard_resource(&err, [](const char *e) { FreeEXRErrorMessage(e); });
            if (auto ret = SaveEXRImageToFile(&image, &header, path_str.c_str(), &err); ret != TINYEXR_SUCCESS) {
                LUISA_EXCEPTION_IF("Failed to save texture as OpenEXR image: ", path);
            }
            LUISA_INFO("Done saving texture: ", path);
        });
    } else if (extension == ".hdr") {
        LUISA_ERROR_IF_NOT(format() == PixelFormat::RGBA32F, "Only RGBA32F textures are allowed to be saved as HDRI images.");
        LUISA_WARNING("Alpha channels will be discarded when textures saved as HDRI images.");
        auto pixels = std::make_shared<std::vector<float>>(pixel_count() * 4);
        copy_to(dispatch, pixels->data());
        dispatch.when_completed([pixels = std::move(pixels), count = pixel_count(), w = width(), h = height(), path_str, path] {
            stbi_write_hdr(path_str.c_str(), w, h, 4, pixels->data());
            LUISA_INFO("Done saving texture: ", path);
        });
    } else {
        LUISA_ERROR_IF_NOT(extension == ".bmp" || extension == ".png" || extension == ".tga" || extension == ".jpg" || extension == ".jpeg",
                           "Failed to save texture with unsupported file extension: ", path);
        auto pixels = std::make_shared<std::vector<uchar>>(pixel_count() * channels());
        copy_to(dispatch, pixels->data());
        dispatch.when_completed([pixels = std::move(pixels), count = pixel_count(), w = width(), h = height(), c = channels(), path_str, path, extension] {
            if (extension == ".bmp") {
                LUISA_WARNING_IF(c == 2, "Saving RG8U textures to Bitmap images may lead to unexpected results.");
                stbi_write_bmp(path_str.c_str(), w, h, c, pixels->data());
            } else if (extension == ".png") {
                stbi_write_png(path_str.c_str(), w, h, c, pixels->data(), 0);
            } else if (extension == ".tga") {
                stbi_write_tga(path_str.c_str(), w, h, c, pixels->data());
            } else {
                LUISA_WARNING_IF(c == 2, "Saving RG8U textures to JPEG images may lead to unexpected results.");
                stbi_write_jpg(path_str.c_str(), w, h, c, pixels->data(), 100);
            }
            LUISA_INFO("Done saving texture: ", path);
        });
    }
}

}
