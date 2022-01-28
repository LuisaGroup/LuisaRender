//
// Created by Mike Smith on 2022/1/15.
//

#include <array>

#include <tinyexr.h>
#include <stb/stb_image.h>

#include <core/logging.h>
#include <util/imageio.h>

namespace luisa::render {

inline LoadedImage LoadedImage::_load_float(const std::filesystem::path &path, storage_type storage) noexcept {
    auto filename = path.string();
    if (storage != storage_type::FLOAT1 &&
        storage != storage_type::FLOAT2 &&
        storage != storage_type::FLOAT4) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid pixel storage {:02x} for FLOAT image '{}'.",
            luisa::to_underlying(storage), filename);
    }
    auto expected_channels = compute::pixel_storage_channel_count(storage);
    auto ext = path.extension();
    if (ext == ".exr") {
        EXRVersion exr_version;
        if (ParseEXRVersionFromFile(&exr_version, filename.c_str()) != TINYEXR_SUCCESS) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "invalid OpenEXR image '{}'.",
                filename);
        }
        if (exr_version.multipart || exr_version.tiled) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "OpenEXR image '{}' is not supported.",
                filename);
        }
        EXRHeader exr_header;
        InitEXRHeader(&exr_header);
        const char *err = nullptr;
        if (ParseEXRHeaderFromFile(
                &exr_header, &exr_version,
                filename.c_str(), &err) != TINYEXR_SUCCESS) [[unlikely]] {
            luisa::string error{"unknown error"};
            if (err) [[likely]] {
                error = err;
                FreeEXRErrorMessage(err);
            }
            LUISA_ERROR_WITH_LOCATION(
                "Failed to parse OpenEXR image '{}': {}.",
                filename, error);
        }
        for (int i = 0; i < exr_header.num_channels; i++) {
            if (exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
                exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
            }
        }
        err = nullptr;
        EXRImage exr_image;
        InitEXRImage(&exr_image);
        if (LoadEXRImageFromFile(
                &exr_image, &exr_header,
                filename.c_str(), &err) != TINYEXR_SUCCESS) [[unlikely]] {
            luisa::string error{"unknown error"};
            if (err) [[likely]] {
                error = err;
                FreeEXRErrorMessage(err);
            }
            FreeEXRHeader(&exr_header);
            LUISA_ERROR_WITH_LOCATION(
                "Failed to load OpenEXR image '{}': {}.",
                filename, error);
        }
        auto width = static_cast<uint>(exr_image.width);
        auto height = static_cast<uint>(exr_image.height);
        auto float_count = width * height * expected_channels;
        auto pixels = luisa::allocate<float>(float_count);
        auto num_channels = static_cast<uint>(exr_image.num_channels);
        if (expected_channels == 1u) {
            if (num_channels != 1u) {
                LUISA_ERROR_WITH_LOCATION(
                    "Expected 1 channel from OpenEXR "
                    "image '{}' with {} channels.",
                    filename, num_channels);
            }
            std::memcpy(
                pixels, exr_image.images[0],
                float_count * sizeof(float));
        } else if (expected_channels == 2u) {
            using namespace std::string_view_literals;
            std::array<uint, 2u> swizzle{};
            if (num_channels == 1u) {
                swizzle = {0u, 0u};
            } else {
                std::array desc{"R"sv, "G"sv};
                std::transform(
                    desc.cbegin(), desc.cend(), swizzle.begin(),
                    [&](auto channel) noexcept {
                        for (auto i = 0u; i < num_channels; i++) {
                            if (exr_header.channels[i].name == channel) {
                                return i;
                            }
                        }
                        LUISA_ERROR_WITH_LOCATION(
                            "Channel '{}' not found in OpenEXR image '{}'.",
                            channel, filename);
                    });
            }
            for (auto i = 0u; i < width * height; i++) {
                for (auto c = 0u; c < 3u; c++) {
                    auto image = exr_image.images[swizzle[c]];
                    pixels[i * 3u + c] = reinterpret_cast<const float *>(image)[i];
                }
            }
        } else if (expected_channels == 4u) {
            std::array<uint, 4u> swizzle{};
            if (num_channels == 1u) {
                swizzle = {0u, 0u, 0u, ~0u};
            } else {
                using namespace std::string_view_literals;
                std::array desc{"R"sv, "G"sv, "B"sv, "A"sv};
                std::transform(
                    desc.cbegin(), desc.cend(), swizzle.begin(),
                    [&](auto channel) noexcept {
                        for (auto i = 0u; i < num_channels; i++) {
                            if (exr_header.channels[i].name == channel) {
                                return i;
                            }
                        }
                        if (channel != "A"sv) {
                            LUISA_ERROR_WITH_LOCATION(
                                "Channel '{}' not found in OpenEXR image '{}'.",
                                channel, filename);
                        }
                        return ~0u;
                    });
            }
            if (swizzle[3] != ~0u) {// has alpha channel
                for (auto i = 0u; i < width * height; i++) {
                    for (auto c = 0u; c < 4u; c++) {
                        auto image = exr_image.images[swizzle[c]];
                        pixels[i * 4u + c] = reinterpret_cast<const float *>(image)[i];
                    }
                }
            } else {
                for (auto i = 0u; i < width * height; i++) {
                    for (auto c = 0u; c < 3u; c++) {
                        auto image = exr_image.images[swizzle[c]];
                        pixels[i * 4u + c] = reinterpret_cast<const float *>(image)[i];
                    }
                    pixels[i * 4u + 3u] = 1.0f;
                }
            }
        }
        FreeEXRImage(&exr_image);
        FreeEXRHeader(&exr_header);
        return {pixels, storage, make_uint2(width, height),
                [](void *p) noexcept {
                    luisa::deallocate(static_cast<float *>(p));
                }};
    }
    if (ext == ".hdr") {
        int w, h, nc;
        auto pixels = stbi_loadf(filename.c_str(), &w, &h, &nc, static_cast<int>(expected_channels));
        if (pixels == nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to load HDR image '{}'.",
                filename);
        }
        return {pixels, storage, make_uint2(w, h),
                [](void *p) noexcept { stbi_image_free(p); }};
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid FLOAT image '{}'.",
        filename);
}

inline LoadedImage LoadedImage::_load_byte(const std::filesystem::path &path, storage_type storage) noexcept {
    auto filename = path.string();
    if (storage != storage_type::BYTE1 &&
        storage != storage_type::BYTE2 &&
        storage != storage_type::BYTE4) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid pixel storage 0x{:02x} for BYTE image '{}'.",
            luisa::to_underlying(storage), filename);
    }
    int w, h, nc;
    auto expected_channels = compute::pixel_storage_channel_count(storage);
    auto pixels = stbi_load(filename.c_str(), &w, &h, &nc, static_cast<int>(expected_channels));
    if (pixels == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load BYTE image '{}'.",
            filename);
    }
    return {pixels, storage, make_uint2(w, h),
            [](void *p) noexcept { stbi_image_free(p); }};
}

LoadedImage LoadedImage::load(const std::filesystem::path &path, LoadedImage::storage_type storage) noexcept {
    switch (storage) {
        case compute::PixelStorage::BYTE1:
        case compute::PixelStorage::BYTE2:
        case compute::PixelStorage::BYTE4:
            return _load_byte(path, storage);
        case compute::PixelStorage::SHORT1:
        case compute::PixelStorage::SHORT2:
        case compute::PixelStorage::SHORT4:
            LUISA_ERROR_WITH_LOCATION("Not implemented.");
        case compute::PixelStorage::INT1:
        case compute::PixelStorage::INT2:
        case compute::PixelStorage::INT4:
            LUISA_ERROR_WITH_LOCATION("Not implemented.");
        case compute::PixelStorage::HALF1:
        case compute::PixelStorage::HALF2:
        case compute::PixelStorage::HALF4:
            LUISA_ERROR_WITH_LOCATION("Not implemented.");
        case compute::PixelStorage::FLOAT1:
        case compute::PixelStorage::FLOAT2:
        case compute::PixelStorage::FLOAT4:
            return _load_float(path, storage);
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid pixel storage: {:02x}.",
        luisa::to_underlying(storage));
}

}// namespace luisa::render
