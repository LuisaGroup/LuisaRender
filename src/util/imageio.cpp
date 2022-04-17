//
// Created by Mike Smith on 2022/1/15.
//

#include <array>

#include <tinyexr.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include <core/logging.h>
#include <util/imageio.h>
#include <util/half.h>

namespace luisa::render {

[[nodiscard]] inline auto parse_exr_header(const char *filename) noexcept {
    EXRVersion exr_version;
    if (ParseEXRVersionFromFile(&exr_version, filename) != TINYEXR_SUCCESS) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "invalid OpenEXR image '{}'.", filename);
    }
    if (exr_version.multipart || exr_version.tiled) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "OpenEXR image '{}' is not supported.", filename);
    }
    EXRHeader exr_header;
    InitEXRHeader(&exr_header);
    const char *err = nullptr;
    if (ParseEXRHeaderFromFile(
            &exr_header, &exr_version,
            filename, &err) != TINYEXR_SUCCESS) [[unlikely]] {
        luisa::string error{"unknown error"};
        if (err) [[likely]] {
            error = err;
            FreeEXRErrorMessage(err);
        }
        LUISA_ERROR_WITH_LOCATION(
            "Failed to parse OpenEXR image '{}': {}",
            filename, error);
    }
    return exr_header;
}

template<typename T>
[[nodiscard]] inline std::pair<void *, uint2> parse_exr_image(const char *filename, EXRHeader &exr_header, uint expected_channels) noexcept {
    for (int i = 0; i < exr_header.num_channels; i++) {
        if constexpr (std::is_same_v<T, float>) {
            exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;
        } else if constexpr (std::is_same_v<T, uint>) {
            exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_UINT;
        } else {
            static_assert(always_false_v<T>);
        }
    }
    const char *err = nullptr;
    EXRImage exr_image;
    InitEXRImage(&exr_image);
    if (LoadEXRImageFromFile(
            &exr_image, &exr_header,
            filename, &err) != TINYEXR_SUCCESS) [[unlikely]] {
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
    auto value_count = width * height * expected_channels;
    auto pixels = luisa::allocate<T>(value_count);
    auto num_channels = static_cast<uint>(exr_image.num_channels);
    if (expected_channels == 1u) {
        if (num_channels != 1u) {
            LUISA_ERROR_WITH_LOCATION(
                "Expected 1 channel from OpenEXR "
                "image '{}' with {} channels.",
                filename, num_channels);
        }
        std::memcpy(pixels, exr_image.images[0], value_count * sizeof(T));
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
                pixels[i * 3u + c] = reinterpret_cast<const T *>(image)[i];
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
                    pixels[i * 4u + c] = reinterpret_cast<const T *>(image)[i];
                }
            }
        } else {
            for (auto i = 0u; i < width * height; i++) {
                for (auto c = 0u; c < 3u; c++) {
                    auto image = exr_image.images[swizzle[c]];
                    pixels[i * 4u + c] = reinterpret_cast<const T *>(image)[i];
                }
                pixels[i * 4u + 3u] = 1.0f;
            }
        }
    }
    FreeEXRImage(&exr_image);
    FreeEXRHeader(&exr_header);
    return std::make_pair(pixels, make_uint2(width, height));
}

template<typename T>
[[nodiscard]] inline auto load_exr(const char *filename, uint expected_channels) noexcept {
    auto exr_header = parse_exr_header(filename);
    return parse_exr_image<T>(filename, exr_header, expected_channels);
}

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
    auto ext = path.extension().string();
    for (auto &c : ext) { c = static_cast<char>(tolower(c)); }
    if (ext == ".exr") {
        auto [pixels, size] = load_exr<float>(filename.c_str(), expected_channels);
        return {
            pixels, storage, size,
            [](void *p) noexcept {
                luisa::deallocate(static_cast<float *>(p));
            }};
    }
    int w, h, nc;
    auto pixels = stbi_loadf(filename.c_str(), &w, &h, &nc, static_cast<int>(expected_channels));
    if (pixels == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load FLOAT image '{}': {}.",
            filename, stbi_failure_reason());
    }
    return {pixels, storage, make_uint2(w, h), stbi_image_free};
}

LoadedImage LoadedImage::_load_half(const std::filesystem::path &path, LoadedImage::storage_type storage) noexcept {
    auto filename = path.string();
    if (storage != storage_type::HALF1 &&
        storage != storage_type::HALF2 &&
        storage != storage_type::HALF4) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid pixel storage {:02x} for HALF image '{}'.",
            luisa::to_underlying(storage), filename);
    }
    auto expected_channels = compute::pixel_storage_channel_count(storage);
    auto ext = path.extension().string();
    for (auto &c : ext) { c = static_cast<char>(tolower(c)); }
    if (ext == ".exr") {
        auto [pixels, size] = load_exr<uint16_t>(filename.c_str(), expected_channels);
        return {
            pixels, storage, size,
            [](void *p) noexcept {
                luisa::deallocate(static_cast<short *>(p));
            }};
    }
    int w, h, nc;
    auto pixels = stbi_loadf(filename.c_str(), &w, &h, &nc, static_cast<int>(expected_channels));
    if (pixels == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load HALF image '{}': {}.",
            filename, stbi_failure_reason());
    }
    auto half_pixels = luisa::allocate<uint16_t>(w * h * expected_channels);
    for (auto i = 0u; i < w * h * expected_channels; i++) {
        half_pixels[i] = float_to_half(
            reinterpret_cast<const float *>(pixels)[i]);
    }
    stbi_image_free(pixels);
    return {
        half_pixels, storage, make_uint2(w, h),
        [](void *p) noexcept {
            luisa::deallocate(static_cast<uint16_t *>(p));
        }};
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
            "Failed to load BYTE image '{}': {}.",
            filename, stbi_failure_reason());
    }
    return {pixels, storage, make_uint2(w, h), stbi_image_free};
}

LoadedImage LoadedImage::_load_short(const std::filesystem::path &path, LoadedImage::storage_type storage) noexcept {
    auto filename = path.string();
    if (storage != storage_type::SHORT1 &&
        storage != storage_type::SHORT2 &&
        storage != storage_type::SHORT4) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid pixel storage 0x{:02x} for SHORT image '{}'.",
            luisa::to_underlying(storage), filename);
    }
    int w, h, nc;
    auto expected_channels = compute::pixel_storage_channel_count(storage);
    auto pixels = stbi_load_16(filename.c_str(), &w, &h, &nc, static_cast<int>(expected_channels));
    if (pixels == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load SHORT image '{}': {}.",
            filename, stbi_failure_reason());
    }
    return {pixels, storage, make_uint2(w, h), stbi_image_free};
}

LoadedImage LoadedImage::_load_int(const std::filesystem::path &path, LoadedImage::storage_type storage) noexcept {
    auto filename = path.string();
    if (storage != storage_type::INT1 &&
        storage != storage_type::INT2 &&
        storage != storage_type::INT4) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid pixel storage {:02x} for INT image '{}'.",
            luisa::to_underlying(storage), filename);
    }
    auto expected_channels = compute::pixel_storage_channel_count(storage);
    auto ext = path.extension().string();
    for (auto &c : ext) { c = static_cast<char>(tolower(c)); }
    if (ext != ".exr") [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid INT image: '{}'.", filename);
    }
    auto [pixels, size] = load_exr<uint>(filename.c_str(), expected_channels);
    return {
        pixels, storage, size,
        [](void *p) noexcept {
            luisa::deallocate(static_cast<uint32_t *>(p));
        }};
}

LoadedImage LoadedImage::load(const std::filesystem::path &path, LoadedImage::storage_type storage) noexcept {
    static std::once_flag flag;
    std::call_once(flag, [] { stbi_ldr_to_hdr_gamma(1.0f); });
    switch (storage) {
        case compute::PixelStorage::BYTE1:
        case compute::PixelStorage::BYTE2:
        case compute::PixelStorage::BYTE4:
            return _load_byte(path, storage);
        case compute::PixelStorage::SHORT1:
        case compute::PixelStorage::SHORT2:
        case compute::PixelStorage::SHORT4:
            return _load_short(path, storage);
        case compute::PixelStorage::INT1:
        case compute::PixelStorage::INT2:
        case compute::PixelStorage::INT4:
            return _load_int(path, storage);
        case compute::PixelStorage::HALF1:
        case compute::PixelStorage::HALF2:
        case compute::PixelStorage::HALF4:
            return _load_half(path, storage);
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

LoadedImage::storage_type LoadedImage::parse_storage(const std::filesystem::path &path) noexcept {
    auto ext = path.extension().string();
    auto path_string = path.string();
    for (auto &c : ext) { c = static_cast<char>(tolower(c)); }
    auto storage = storage_type::FLOAT4;
    auto size = make_uint2();
    if (ext == ".exr") {
        auto exr_header = parse_exr_header(path_string.c_str());
        if (auto t = exr_header.pixel_types[0];
            t == TINYEXR_PIXELTYPE_UINT) {
            if (exr_header.num_channels == 1u) {
                storage = storage_type::INT1;
            } else if (exr_header.num_channels == 2u) {
                storage = storage_type::INT2;
            } else {
                storage = storage_type::INT4;
            }
        } else if (t == TINYEXR_PIXELTYPE_HALF) {
            if (exr_header.num_channels == 1u) {
                storage = storage_type::HALF1;
            } else if (exr_header.num_channels == 2u) {
                storage = storage_type::HALF2;
            } else {
                storage = storage_type::HALF4;
            }
        } else {
            if (exr_header.num_channels == 1u) {
                storage = storage_type::FLOAT1;
            } else if (exr_header.num_channels == 2u) {
                storage = storage_type::FLOAT2;
            }
        }
        FreeEXRHeader(&exr_header);
    } else if (ext == ".hdr") {
        storage = storage_type::HALF4;
    } else {
        auto p = path_string.c_str();
        auto file = fopen(p, "rb");
        if (file == nullptr) {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to open image '{}'.",
                path_string);
        }
        auto width = 0, height = 0, channels = 0;
        if (!stbi_info_from_file(file, &width, &height, &channels)) [[unlikely]] {
            fclose(file);
            LUISA_ERROR_WITH_LOCATION(
                "Failed to parse info from image '{}': {}.",
                path_string, stbi_failure_reason());
        }
        if (stbi_is_16_bit_from_file(file)) {
            if (channels == 1) {
                storage = storage_type::SHORT1;
            } else if (channels == 2) {
                storage = storage_type::SHORT2;
            } else {
                storage = storage_type::SHORT4;
            }
        } else {
            if (channels == 1) {
                storage = storage_type::BYTE1;
            } else if (channels == 2) {
                storage = storage_type::BYTE2;
            } else {
                storage = storage_type::BYTE4;
            }
        }
        fclose(file);
    }
    return storage;
}

LoadedImage LoadedImage::load(const std::filesystem::path &path) noexcept {
    auto ext = path.extension().string();
    auto path_string = path.string();
    for (auto &c : ext) { c = static_cast<char>(tolower(c)); }
    if (ext == ".exr") {
        auto exr_header = parse_exr_header(path_string.c_str());
        auto t = exr_header.pixel_types[0];
        auto load_image = [&exr_header, t, p = path_string.c_str()]() noexcept {
            if (t == TINYEXR_PIXELTYPE_UINT) {
                auto expected_channels = 4u;
                auto storage = storage_type::INT4;
                if (exr_header.num_channels == 1u) {
                    expected_channels = 1u;
                    storage = storage_type::INT1;
                } else if (exr_header.num_channels == 2u) {
                    expected_channels = 2u;
                    storage = storage_type::INT2;
                }
                auto [pixels, size] = parse_exr_image<uint>(
                    p, exr_header, expected_channels);
                return std::make_tuple(
                    pixels, size, storage,
                    luisa::function<void(void *)>{[](void *p) noexcept {
                        luisa::deallocate(static_cast<uint *>(p));
                    }});
            }
            if (t == TINYEXR_PIXELTYPE_HALF) {
                auto expected_channels = 4u;
                auto storage = storage_type::HALF4;
                if (exr_header.num_channels == 1u) {
                    expected_channels = 1u;
                    storage = storage_type::HALF1;
                } else if (exr_header.num_channels == 2u) {
                    expected_channels = 2u;
                    storage = storage_type::HALF2;
                }
                auto [pixels, size] = parse_exr_image<uint16_t>(
                    p, exr_header, expected_channels);
                return std::make_tuple(
                    pixels, size, storage,
                    luisa::function<void(void *)>{[](void *p) noexcept {
                        luisa::deallocate(static_cast<uint16_t *>(p));
                    }});
            }
            auto expected_channels = 4u;
            auto storage = storage_type::FLOAT4;
            if (exr_header.num_channels == 1u) {
                expected_channels = 1u;
                storage = storage_type::FLOAT1;
            } else if (exr_header.num_channels == 2u) {
                expected_channels = 2u;
                storage = storage_type::FLOAT2;
            }
            auto [pixels, size] = parse_exr_image<float>(
                p, exr_header, expected_channels);
            return std::make_tuple(
                pixels, size, storage,
                luisa::function<void(void *)>{[](void *p) noexcept {
                    luisa::deallocate(static_cast<float *>(p));
                }});
        };
        auto [pixels, size, storage, deleter] = load_image();
        return {pixels, storage, size, std::move(deleter)};
    }
    if (ext == ".hdr") {
        return load(path, storage_type::HALF4);
    }
    auto p = path_string.c_str();
    auto file = fopen(p, "rb");
    if (file == nullptr) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to open image '{}'.",
            path_string);
    }
    auto width = 0, height = 0, channels = 0;
    if (!stbi_info_from_file(file, &width, &height, &channels)) [[unlikely]] {
        fclose(file);
        LUISA_ERROR_WITH_LOCATION(
            "Failed to parse info from image '{}': {}.",
            path_string, stbi_failure_reason());
    }
    if (stbi_is_16_bit_from_file(file)) {
        auto expected_channels = 4;
        auto storage = storage_type::SHORT4;
        if (channels == 1) {
            expected_channels = 1;
            storage = storage_type::SHORT1;
        } else if (channels == 2) {
            expected_channels = 2;
            storage = storage_type::SHORT2;
        }
        auto pixels = stbi_load_from_file_16(
            file, &width, &height, &channels, expected_channels);
        fclose(file);
        if (pixels == nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to load image '{}': {}.",
                path_string, stbi_failure_reason());
        }
        return {pixels, storage, make_uint2(width, height), stbi_image_free};
    }
    auto expected_channels = 4;
    auto storage = storage_type::BYTE4;
    if (channels == 1) {
        expected_channels = 1;
        storage = storage_type::BYTE1;
    } else if (channels == 2) {
        expected_channels = 2;
        storage = storage_type::BYTE2;
    }
    auto pixels = stbi_load_from_file(
        file, &width, &height, &channels, expected_channels);
    fclose(file);
    if (pixels == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load image '{}': {}.",
            path_string, stbi_failure_reason());
    }
    return {pixels, storage, make_uint2(width, height), stbi_image_free};
}

void save_image(std::filesystem::path path, const float *pixels, uint2 resolution) noexcept {
    // save results
    auto pixel_count = resolution.x * resolution.y;
    auto size = make_int2(resolution);

    if (path.extension() != ".exr" && path.extension() != ".hdr") [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Unexpected film file extension. "
            "Changing to '.exr'.");
        path.replace_extension(".exr");
    }

    if (path.extension() == ".exr") {
        const char *err = nullptr;
        SaveEXR(reinterpret_cast<const float *>(pixels),
                size.x, size.y, 4, false, path.string().c_str(), &err);
        if (err != nullptr) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to save film to '{}'.",
                path.string());
        }
    } else if (path.extension() == ".hdr") {
        stbi_write_hdr(path.string().c_str(), size.x, size.y, 4, reinterpret_cast<const float *>(pixels));
    }
}

}// namespace luisa::render
