//
// Created by Mike Smith on 2022/1/15.
//

#include <array>

#include <tinyexr.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image_resize.h>

#include <core/logging.h>
#include <util/imageio.h>
#include <util/half.h>

namespace luisa::render {

void LoadedImage::_destroy() noexcept {
    if (*this) { _deleter(_pixels); }
}

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
    auto pixels = luisa::allocate_with_allocator<T>(value_count);
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
            for (auto channel = 0u; channel < desc.size(); channel++) {
                auto found = false;
                for (auto i = 0u; i < num_channels; i++) {
                    if (exr_header.channels[i].name == desc[channel]) {
                        swizzle[channel] = i;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    LUISA_ERROR_WITH_LOCATION(
                        "Channel '{}' not found in OpenEXR image '{}'.",
                        channel, filename);
                }
            }
        }
        for (auto i = 0u; i < width * height; i++) {
            for (auto c = 0u; c < 2u; c++) {
                auto image = exr_image.images[swizzle[c]];
                pixels[i * 2u + c] = reinterpret_cast<const T *>(image)[i];
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
                luisa::deallocate_with_allocator(static_cast<float *>(p));
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
                luisa::deallocate_with_allocator(static_cast<short *>(p));
            }};
    }
    int w, h, nc;
    auto pixels = stbi_loadf(filename.c_str(), &w, &h, &nc, static_cast<int>(expected_channels));
    if (pixels == nullptr) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to load HALF image '{}': {}.",
            filename, stbi_failure_reason());
    }
    auto half_pixels = luisa::allocate_with_allocator<uint16_t>(w * h * expected_channels);
    for (auto i = 0u; i < w * h * expected_channels; i++) {
        half_pixels[i] = float_to_half(
            reinterpret_cast<const float *>(pixels)[i]);
    }
    stbi_image_free(pixels);
    return {
        half_pixels, storage, make_uint2(w, h),
        [](void *p) noexcept {
            luisa::deallocate_with_allocator(static_cast<uint16_t *>(p));
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
            luisa::deallocate_with_allocator(static_cast<uint32_t *>(p));
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
                        luisa::deallocate_with_allocator(static_cast<uint *>(p));
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
                        luisa::deallocate_with_allocator(static_cast<uint16_t *>(p));
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
                    luisa::deallocate_with_allocator(static_cast<float *>(p));
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

LoadedImage LoadedImage::create(uint2 resolution, LoadedImage::storage_type storage) noexcept {
    auto size_bytes = pixel_storage_size(storage, make_uint3(resolution.x, resolution.y, 1u));
    auto pixels = luisa::allocate_with_allocator<std::byte>(size_bytes);
    return {pixels, storage, resolution, luisa::function<void(void *)>{[](void *p) noexcept {
                luisa::deallocate_with_allocator(static_cast<std::byte *>(p));
            }}};
}

LoadedImage &LoadedImage::operator=(LoadedImage &&rhs) noexcept {
    if (&rhs != this) [[likely]] {
        _destroy();
        _pixels = rhs._pixels;
        _resolution = rhs._resolution;
        _storage = rhs._storage;
        _deleter = std::move(rhs._deleter);
        rhs._pixels = nullptr;
    }
    return *this;
}

LoadedImage::LoadedImage(LoadedImage &&another) noexcept
    : _pixels{another._pixels},
      _resolution{another._resolution},
      _storage{another._storage},
      _deleter{std::move(another._deleter)} { another._pixels = nullptr; }

LoadedImage::~LoadedImage() noexcept { _destroy(); }

LoadedImage::LoadedImage(void *pixels,
                         LoadedImage::storage_type storage,
                         uint2 resolution,
                         luisa::function<void(void *)> deleter) noexcept
    : _pixels{pixels}, _resolution{resolution},
      _storage{storage}, _deleter{std::move(deleter)} {}

//float4 LoadedImage::read(uint2 p) const noexcept {
//    auto i = p.x + p.y * _resolution.x;
//    constexpr auto byte_to_float = [](auto x) noexcept { return static_cast<float>(x) * (1.f / 255.f); };
//    constexpr auto short_to_float = [](auto x) noexcept { return static_cast<float>(x) * (1.f / 65535.f); };
//    switch (_storage) {
//        case compute::PixelStorage::BYTE1:
//            return make_float4(byte_to_float(static_cast<const uint8_t *>(_pixels)[i]),
//                               byte_to_float(static_cast<const uint8_t *>(_pixels)[i]),
//                               byte_to_float(static_cast<const uint8_t *>(_pixels)[i]), 1.f);
//        case compute::PixelStorage::BYTE2:
//            return make_float4(byte_to_float(static_cast<const uint8_t *>(_pixels)[i * 2u + 0u]),
//                               byte_to_float(static_cast<const uint8_t *>(_pixels)[i * 2u + 1u]), 0.f, 1.f);
//        case compute::PixelStorage::BYTE4:
//            return make_float4(byte_to_float(static_cast<const uint8_t *>(_pixels)[i * 4u + 0u]),
//                               byte_to_float(static_cast<const uint8_t *>(_pixels)[i * 4u + 1u]),
//                               byte_to_float(static_cast<const uint8_t *>(_pixels)[i * 4u + 2u]),
//                               byte_to_float(static_cast<const uint8_t *>(_pixels)[i * 4u + 3u]));
//        case compute::PixelStorage::SHORT1:
//            return make_float4(short_to_float(static_cast<const uint16_t *>(_pixels)[i]),
//                               short_to_float(static_cast<const uint16_t *>(_pixels)[i]),
//                               short_to_float(static_cast<const uint16_t *>(_pixels)[i]), 1.f);
//        case compute::PixelStorage::SHORT2:
//            return make_float4(short_to_float(static_cast<const uint16_t *>(_pixels)[i * 2u + 0u]),
//                               short_to_float(static_cast<const uint16_t *>(_pixels)[i * 2u + 1u]), 0.f, 0.f);
//        case compute::PixelStorage::SHORT4:
//            return make_float4(short_to_float(static_cast<const uint16_t *>(_pixels)[i * 4u + 0u]),
//                               short_to_float(static_cast<const uint16_t *>(_pixels)[i * 4u + 1u]),
//                               short_to_float(static_cast<const uint16_t *>(_pixels)[i * 4u + 2u]),
//                               short_to_float(static_cast<const uint16_t *>(_pixels)[i * 4u + 3u]));
//        case compute::PixelStorage::HALF1:
//            return make_float4(half_to_float(static_cast<const uint16_t *>(_pixels)[i]),
//                               half_to_float(static_cast<const uint16_t *>(_pixels)[i]),
//                               half_to_float(static_cast<const uint16_t *>(_pixels)[i]), 1.f);
//        case compute::PixelStorage::HALF2:
//            return make_float4(half_to_float(static_cast<const uint16_t *>(_pixels)[i * 2u + 0u]),
//                               half_to_float(static_cast<const uint16_t *>(_pixels)[i * 2u + 1u]), 0.f, 0.f);
//        case compute::PixelStorage::HALF4:
//            return make_float4(half_to_float(static_cast<const uint16_t *>(_pixels)[i * 4u + 0u]),
//                               half_to_float(static_cast<const uint16_t *>(_pixels)[i * 4u + 1u]),
//                               half_to_float(static_cast<const uint16_t *>(_pixels)[i * 4u + 2u]),
//                               half_to_float(static_cast<const uint16_t *>(_pixels)[i * 4u + 3u]));
//        case compute::PixelStorage::FLOAT1:
//            return make_float4(static_cast<const float *>(_pixels)[i],
//                               static_cast<const float *>(_pixels)[i],
//                               static_cast<const float *>(_pixels)[i], 1.f);
//        case compute::PixelStorage::FLOAT2:
//            return make_float4(static_cast<const float *>(_pixels)[i * 2u + 0u],
//                               static_cast<const float *>(_pixels)[i * 2u + 1u], 0.f, 0.f);
//        case compute::PixelStorage::FLOAT4:
//            return make_float4(static_cast<const float *>(_pixels)[i * 4u + 0u],
//                               static_cast<const float *>(_pixels)[i * 4u + 1u],
//                               static_cast<const float *>(_pixels)[i * 4u + 2u],
//                               static_cast<const float *>(_pixels)[i * 4u + 3u]);
//        default: break;
//    }
//    return make_float4();
//}
//
//void LoadedImage::write(uint2 p, float4 v) noexcept {
//    auto i = p.x + p.y * _resolution.x;
//    constexpr auto float_to_byte = [](auto x) noexcept { return static_cast<uint8_t>(std::clamp(std::round(x * 255.f), 0.f, 255.f)); };
//    constexpr auto float_to_short = [](auto x) noexcept { return static_cast<uint16_t>(std::clamp(std::round(x * 65535.f), 0.f, 65535.f)); };
//    switch (_storage) {
//        case compute::PixelStorage::BYTE1:
//            static_cast<uint8_t *>(_pixels)[i] = float_to_byte(v.x);
//            break;
//        case compute::PixelStorage::BYTE2:
//            static_cast<uint8_t *>(_pixels)[i * 2u + 0u] = float_to_byte(v.x);
//            static_cast<uint8_t *>(_pixels)[i * 2u + 1u] = float_to_byte(v.y);
//            break;
//        case compute::PixelStorage::BYTE4:
//            static_cast<uint8_t *>(_pixels)[i * 4u + 0u] = float_to_byte(v.x);
//            static_cast<uint8_t *>(_pixels)[i * 4u + 1u] = float_to_byte(v.y);
//            static_cast<uint8_t *>(_pixels)[i * 4u + 2u] = float_to_byte(v.y);
//            static_cast<uint8_t *>(_pixels)[i * 4u + 3u] = float_to_byte(v.z);
//            break;
//        case compute::PixelStorage::SHORT1:
//            static_cast<uint16_t *>(_pixels)[i] = float_to_short(v.x);
//            break;
//        case compute::PixelStorage::SHORT2:
//            static_cast<uint16_t *>(_pixels)[i * 2u + 0u] = float_to_short(v.x);
//            static_cast<uint16_t *>(_pixels)[i * 2u + 1u] = float_to_short(v.y);
//            break;
//        case compute::PixelStorage::SHORT4:
//            static_cast<uint16_t *>(_pixels)[i * 4u + 0u] = float_to_short(v.x);
//            static_cast<uint16_t *>(_pixels)[i * 4u + 1u] = float_to_short(v.y);
//            static_cast<uint16_t *>(_pixels)[i * 4u + 2u] = float_to_short(v.z);
//            static_cast<uint16_t *>(_pixels)[i * 4u + 3u] = float_to_short(v.w);
//            break;
//        case compute::PixelStorage::HALF1:
//            static_cast<uint16_t *>(_pixels)[i] = float_to_half(v.x);
//            break;
//        case compute::PixelStorage::HALF2:
//            static_cast<uint16_t *>(_pixels)[i * 2u + 0u] = float_to_half(v.x);
//            static_cast<uint16_t *>(_pixels)[i * 2u + 1u] = float_to_half(v.y);
//            break;
//        case compute::PixelStorage::HALF4:
//            static_cast<uint16_t *>(_pixels)[i * 4u + 0u] = float_to_half(v.x);
//            static_cast<uint16_t *>(_pixels)[i * 4u + 1u] = float_to_half(v.y);
//            static_cast<uint16_t *>(_pixels)[i * 4u + 2u] = float_to_half(v.z);
//            static_cast<uint16_t *>(_pixels)[i * 4u + 3u] = float_to_half(v.w);
//            break;
//        case compute::PixelStorage::FLOAT1:
//            static_cast<float *>(_pixels)[i] = v.x;
//            break;
//        case compute::PixelStorage::FLOAT2:
//            static_cast<float *>(_pixels)[i * 2u + 0u] = v.x;
//            static_cast<float *>(_pixels)[i * 2u + 1u] = v.y;
//            break;
//        case compute::PixelStorage::FLOAT4:
//            static_cast<float *>(_pixels)[i * 4u + 0u] = v.x;
//            static_cast<float *>(_pixels)[i * 4u + 1u] = v.y;
//            static_cast<float *>(_pixels)[i * 4u + 2u] = v.z;
//            static_cast<float *>(_pixels)[i * 4u + 3u] = v.w;
//            break;
//        default: break;
//    }
//}

void save_image(std::filesystem::path path, const float *pixels, uint2 resolution, uint components) noexcept {
    // save results
    auto pixel_count = resolution.x * resolution.y;
    auto size = make_int2(resolution);
    auto ext = path.extension().string();
    for (auto &c : ext) { c = static_cast<char>(std::tolower(c)); }
    if (ext != ".exr" && ext != ".hdr") [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Unsupported image extension '{}' in path '{}'. "
            "Falling back to '.exr'.",
            ext, path.string());
        path.replace_extension(".exr");
    }

    auto c = static_cast<int>(std::clamp(components, 1u, 4u));
    if (ext == ".exr") {
        const char *err = nullptr;
        SaveEXR(reinterpret_cast<const float *>(pixels),
                size.x, size.y, c, false, path.string().c_str(), &err);
        if (err != nullptr) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to save film to '{}': {}.",
                path.string(), err);
        }
    } else if (ext == ".hdr") {
        if (!stbi_write_hdr(path.string().c_str(), size.x, size.y,
                            c, reinterpret_cast<const float *>(pixels))) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to save film to '{}': {}.",
                path.string(), stbi_failure_reason());
        }
    }
}

void save_image(std::filesystem::path path, const uint8_t *pixels, uint2 resolution, uint components) noexcept {
    LUISA_INFO("Saving image ({}x{}x{}) to '{}'.",
               resolution.x, resolution.y, components, path.string());
    auto ext = path.extension().string();
    for (auto &c : ext) { c = static_cast<char>(std::tolower(c)); }
    if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" &&
        ext != ".bmp" && ext != ".tga") [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Unsupported image extension '{}' in path '{}'. "
            "Falling back to '.png'.",
            ext, path.string());
        path.replace_extension(".png");
    }
    auto w = static_cast<int>(resolution.x);
    auto h = static_cast<int>(resolution.y);
    auto c = static_cast<int>(std::clamp(components, 1u, 4u));
    auto p = path.string();
    if (ext == ".png") {
        if (!stbi_write_png(p.c_str(), w, h, c, pixels, 0)) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to save image to '{}': {}.",
                p, stbi_failure_reason());
        }
    } else if (ext == ".jpg" || ext == ".jpeg") {
        if (!stbi_write_jpg(p.c_str(), w, h, c, pixels, 100)) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to save image to '{}': {}.",
                p, stbi_failure_reason());
        }
    } else if (ext == "bmp") {
        if (!stbi_write_bmp(p.c_str(), w, h, c, pixels)) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to save image to '{}': {}.",
                p, stbi_failure_reason());
        }
    } else if (ext == "tga") {
        if (!stbi_write_tga(p.c_str(), w, h, c, pixels)) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to save image to '{}': {}.",
                p, stbi_failure_reason());
        }
    }
}

}// namespace luisa::render
