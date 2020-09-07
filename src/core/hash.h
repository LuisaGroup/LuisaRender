//
// Created by Mike Smith on 2020/8/19.
//

#pragma once

#include <string>
#include <string_view>

#include <core/platform.h>

namespace luisa {
inline namespace utility {

class SHA1 {

public:
    using Digest = std::array<uint32_t, 5>;

private:
    Digest _digest{};
    std::string _buffer;
    uint64_t _transforms{};

public:
    explicit SHA1(const std::string &s);
    [[nodiscard]] auto digest() const noexcept { return _digest; }
};

inline SHA1::Digest sha1_digest(const std::string &s) noexcept {
    return SHA1{s}.digest();
}

inline uint64_t murmur_hash_64a(const void *key, uint32_t len, uint64_t seed) {
    
    constexpr uint64_t m = 0xc6a4a7935bd1e995ull;
    constexpr uint32_t r = 47u;
    
    auto h = seed ^(len * m);
    
    auto data = static_cast<const uint64_t *>(key);
    auto end = data + (len / 8);
    
    while (data != end) {
        uint64_t k = *data++;
        
        k *= m;
        k ^= k >> r;
        k *= m;
        
        h ^= k;
        h *= m;
    }
    
    auto data2 = reinterpret_cast<const unsigned char *>(data);
    
    switch (len & 7u) {
        case 7u:
            h ^= uint64_t(data2[6]) << 48u;
        case 6u:
            h ^= uint64_t(data2[5]) << 40u;
        case 5u:
            h ^= uint64_t(data2[4]) << 32u;
        case 4u:
            h ^= uint64_t(data2[3]) << 24u;
        case 3u:
            h ^= uint64_t(data2[2]) << 16u;
        case 2u:
            h ^= uint64_t(data2[1]) << 8u;
        case 1u:
            h ^= uint64_t(data2[0]);
            h *= m;
        default:
            break;
    };
    
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    
    return h;
}

}}
