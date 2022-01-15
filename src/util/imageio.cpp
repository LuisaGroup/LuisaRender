//
// Created by Mike Smith on 2022/1/15.
//

#include <array>

#include <tinyexr.h>
#include <stb/stb_image.h>

#include <core/logging.h>
#include <util/imageio.h>

namespace luisa::render {

LoadedImage<float> load_hdr_image(const std::filesystem::path &path, uint expected_channels) noexcept {
    auto filename = path.string();
    if (expected_channels != 1u &&
        expected_channels != 3u &&
        expected_channels != 4u) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid expected channel count {} "
            "for HDR image '{}'.",
            expected_channels, filename);
    }
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
        } else if (expected_channels == 3u) {
            using namespace std::string_view_literals;
            std::array<uint, 3u> swizzle{};
            if (num_channels == 1u) {
                swizzle = {0u, 0u, 0u};
            } else {
                std::array desc{"R"sv, "G"sv, "B"sv};
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
        return {pixels, make_uint2(width, height), expected_channels,
                static_cast<typename LoadedImage<float>::deleter_type>(
                    [](float *p) noexcept { luisa::deallocate(p); })};
    }
    if (ext == ".hdr") {
        int w, h, nc;
        auto pixels = stbi_loadf(filename.c_str(), &w, &h, &nc, static_cast<int>(expected_channels));
        if (pixels == nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to load HDR image '{}'.",
                filename);
        }
        return {pixels, make_uint2(w, h), static_cast<uint>(nc),
                static_cast<typename LoadedImage<float>::deleter_type>(
                    [](float *p) noexcept { stbi_image_free(p); })};
    }
    LUISA_ERROR_WITH_LOCATION(
        "Invalid HDR image '{}'.",
        filename);
}

LoadedImage<uint8_t> load_ldr_image(const std::filesystem::path &path, uint expected_channels) noexcept {
    auto filename = path.string();
    if (expected_channels != 1u &&
        expected_channels != 3u &&
        expected_channels != 4u) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid expected channel count {} "
            "for HDR image '{}'.",
            expected_channels, filename);
    }
    int w, h, nc;
    auto pixels = stbi_load(filename.c_str(), &w, &h, &nc, static_cast<int>(expected_channels));
    if (pixels == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load LDR image '{}'.",
            filename);
    }
    return {pixels, make_uint2(w, h), static_cast<uint>(nc),
            static_cast<typename LoadedImage<uint8_t>::deleter_type>(
                [](uint8_t *p) noexcept { stbi_image_free(p); })};
}

}// namespace luisa::render
