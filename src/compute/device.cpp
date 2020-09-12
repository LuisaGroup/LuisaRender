//
// Created by Mike Smith on 2020/8/8.
//

#include <tinyexr/tinyexr.h>
#include <stb/stb_image.h>
#include "device.h"

namespace luisa::compute {

std::unique_ptr<Device> Device::create(Context *context, uint32_t selection_id) {
    auto &&devices = context->devices();
    if (devices.empty()) {// enumerate available devices
        LUISA_WARNING("Compute device is not specified, enumerating automatically...");
        for (auto backend : {"cuda", "metal"}) {
            try {
                LUISA_INFO("Trying to create device \"", backend, ":", 0u, "\"...");
                auto create_device = context->load_dynamic_function<DeviceCreator>(context->runtime_path("bin") / "backends", serialize("luisa-backend-", backend), "create");
                return std::unique_ptr<Device>{create_device(context, 0u)};
            } catch (const std::exception &e) {
                LUISA_INFO("Failed to create device \"", backend, ":", 0u, "\".");
            }
        }
        LUISA_ERROR("No available compute device found.");
    }
    LUISA_ERROR_IF_NOT(selection_id < devices.size(), "Invalid device selection index: ", selection_id, ", max index is ", devices.size() - 1u, ".");
    auto &&selection = devices[selection_id];
    auto create_device = context->load_dynamic_function<DeviceCreator>(context->runtime_path("bin") / "backends", "luisa-backend-" + selection.backend_name, "create");
    return std::unique_ptr<Device>{create_device(context, selection.device_id)};
}

TextureView Device::load_texture(const std::filesystem::path &file_name, bool gray_to_rgba) {
    
    auto path_str = std::filesystem::canonical(file_name).string();
    auto extension = to_lower(file_name.extension().string());
    if (extension == ".exr") {
        
        // Parse OpenEXR
        EXRVersion exr_version;
        if (auto ret = ParseEXRVersionFromFile(&exr_version, path_str.c_str()); ret != 0) {
            LUISA_EXCEPTION("Failed to parse OpenEXR version for file: ", file_name);
        }
        
        if (exr_version.multipart) {
            LUISA_EXCEPTION("Multipart OpenEXR format is not supported in file: ", file_name);
        }
        
        // 2. Read EXR header
        EXRHeader exr_header;
        InitEXRHeader(&exr_header);
        auto exr_header_guard = guard_resource(&exr_header, [](EXRHeader &h) noexcept { FreeEXRHeader(&h); });
        const char *err = nullptr; // or `nullptr` in C++11 or later.
        auto err_guard = guard_resource(&err, [](const char *e) noexcept { if (e != nullptr) { FreeEXRErrorMessage(e); }});
        if (auto ret = ParseEXRHeaderFromFile(&exr_header, &exr_version, path_str.c_str(), &err); ret != 0) {
            LUISA_EXCEPTION("Failed to parse ", file_name, ": ", err);
        }
        
        if (exr_header.num_channels > 4 || exr_header.tiled ||
            std::any_of(exr_header.channels, exr_header.channels + exr_header.num_channels, [](const EXRChannelInfo &channel) noexcept {
                return channel.pixel_type != TINYEXR_PIXELTYPE_FLOAT && channel.pixel_type != TINYEXR_PIXELTYPE_HALF;
            })) { LUISA_EXCEPTION("Unsupported pixel format in file: ", file_name); }
        
        // Read HALF channel as FLOAT.
        for (int i = 0; i < exr_header.num_channels; i++) {
            if (exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
                exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
            }
        }
        
        EXRImage exr_image;
        InitEXRImage(&exr_image);
        auto exr_image_guard = guard_resource(&exr_image, [](EXRImage &img) mutable { FreeEXRImage(&img); });
        if (auto ret = LoadEXRImageFromFile(&exr_image, &exr_header, path_str.c_str(), &err); ret != 0) {
            LUISA_EXCEPTION("Failed to load ", file_name, ": ", err);
        }
        
        if (exr_image.num_channels == 1) {
            if (gray_to_rgba) {
                auto texture = allocate_texture<float4>(exr_image.width, exr_image.height);
                std::vector<float4> pixels(exr_image.width * exr_image.height);
                for (auto i = 0u; i < exr_image.width * exr_image.height; i++) {
                    pixels[i] = make_float4(
                        reinterpret_cast<float *>(exr_image.images[0])[i],
                        reinterpret_cast<float *>(exr_image.images[0])[i],
                        reinterpret_cast<float *>(exr_image.images[0])[i],
                        1.0f);
                }
                launch(texture.copy_from(pixels.data()), [file_name] { LUISA_INFO("Loaded ", file_name, " as RGBA32F texture."); });
                return texture;
            }
            auto texture = allocate_texture<float>(exr_image.width, exr_image.height);
            launch(texture.copy_from(exr_image.images[0]), [file_name]() mutable {
                LUISA_INFO("Loaded ", file_name, " as float texture.");
            });
            return texture;
        } else if (exr_image.num_channels == 2) {
            auto texture = allocate_texture<float2>(exr_image.width, exr_image.height);
            std::vector<float2> pixels(exr_image.width * exr_image.height);
            for (auto i = 0u; i < exr_image.width * exr_image.height; i++) {
                pixels[i] = make_float2(
                    reinterpret_cast<float *>(exr_image.images[1])[i],
                    reinterpret_cast<float *>(exr_image.images[0])[i]);
            }
            launch(texture.copy_from(pixels.data()), [file_name] { LUISA_INFO("Loaded ", file_name, " as RG32F texture."); });
            return texture;
        } else if (exr_image.num_channels == 3) {
            auto texture = allocate_texture<float4>(exr_image.width, exr_image.height);
            std::vector<float4> pixels(exr_image.width * exr_image.height);
            for (auto i = 0u; i < exr_image.width * exr_image.height; i++) {
                pixels[i] = make_float4(
                    reinterpret_cast<float *>(exr_image.images[2])[i],
                    reinterpret_cast<float *>(exr_image.images[1])[i],
                    reinterpret_cast<float *>(exr_image.images[0])[i],
                    1.0f);
            }
            launch(texture.copy_from(pixels.data()), [file_name] { LUISA_INFO("Loaded ", file_name, " as RGBA32F texture."); });
            return texture;
        } else {
            auto texture = allocate_texture<float4>(exr_image.width, exr_image.height);
            std::vector<float4> pixels(exr_image.width * exr_image.height);
            for (auto i = 0u; i < exr_image.width * exr_image.height; i++) {
                pixels[i] = make_float4(
                    reinterpret_cast<float *>(exr_image.images[3])[i],
                    reinterpret_cast<float *>(exr_image.images[2])[i],
                    reinterpret_cast<float *>(exr_image.images[1])[i],
                    reinterpret_cast<float *>(exr_image.images[0])[i]);
            }
            launch(texture.copy_from(pixels.data()), [file_name] { LUISA_INFO("Loaded ", file_name, " as RGBA32F texture."); });
            return texture;
        }
    } else if (stbi_is_hdr(path_str.c_str())) {
        auto width = 0;
        auto height = 0;
        auto channels = 0;
        auto pixels = stbi_loadf(path_str.c_str(), &width, &height, &channels, 4);
        auto pixels_guard = guard_resource(&pixels, [](float *p) mutable { stbi_image_free(p); });
        LUISA_ERROR_IF(pixels == nullptr, "Failed to load ", file_name);
        auto texture = allocate_texture<float4>(width, height);
        launch(texture.copy_from(pixels), [file_name] { LUISA_INFO("Loaded ", file_name, " as RGBA32F texture."); });
        return texture;
    }
    
    // LDR
    auto width = 0;
    auto height = 0;
    auto channels = 0;
    LUISA_ERROR_IF(stbi_info(path_str.c_str(), &width, &height, &channels) == 0, "Failed to load ", file_name, ": ", stbi_failure_reason());
    auto pixels = stbi_load(path_str.c_str(), &width, &height, &channels, channels == 3 || gray_to_rgba ? 4 : 0);
    auto pixels_guard = guard_resource(&pixels, [](uchar *p) mutable { stbi_image_free(p); });
    if (channels == 1) {
        auto texture = gray_to_rgba ? allocate_texture<uchar4>(width, height) : allocate_texture<uchar>(width, height);
        launch(texture.copy_from(pixels), [file_name, gray_to_rgba] { LUISA_INFO("Loaded ", file_name, gray_to_rgba ? " as RGBA8U texture." : " as R8U texture."); });
        return texture;
    } else if (channels == 2) {
        auto texture = allocate_texture<uchar2>(width, height);
        launch(texture.copy_from(pixels), [file_name] { LUISA_INFO("Loaded ", file_name, " as RG8U texture."); });
        return texture;
    }
    
    auto texture = allocate_texture<uchar4>(width, height);
    launch(texture.copy_from(pixels), [file_name] { LUISA_INFO("Loaded ", file_name, " as RGBA8U texture."); });
    return texture;
}

}// namespace luisa::compute
