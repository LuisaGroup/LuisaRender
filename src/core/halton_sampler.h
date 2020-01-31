////
//// Created by Mike Smith on 2019/12/11.
////
//
//#pragma once
//
//#include "data_types.h"
//#include "prime_numbers.h"
//
//#ifndef LUISA_DEVICE_COMPATIBLE
//#include <cstdint>
//#endif
//
//namespace luisa {
//
//LUISA_CONSTANT_SPACE constexpr auto HALTON_SAMPLER_PATCH_SIZE = 240u;
//LUISA_CONSTANT_SPACE constexpr auto HALTON_SAMPLER_PERMUTATION_TABLE_SIZE = prime_numbers[PRIME_NUMBER_COUNT - 1u] + prime_number_prefix_sums[PRIME_NUMBER_COUNT - 1u];
//
//LUISA_CONSTANT_SPACE constexpr auto ONE_MINUS_EPSILON = 0.99999994f;
//
//struct HaltonSamplerParameters {
//    uint2 frame_size;
//    uint16_t digit_permutations[HALTON_SAMPLER_PERMUTATION_TABLE_SIZE];
//    uint32_t base2_scale;
//    uint32_t base3_scale;
//    uint32_t base2_exponent;
//    uint32_t base3_exponent;
//    uint32_t sample_stride;
//    uint32_t multiplicative_inverses[2];
//
//#ifndef DEVICE_COMPATIBLE
//    HaltonSamplerParameters() noexcept = default;
//#endif
//
//};
//
//struct HaltonSamplerState {
//    uint32_t index_part0;
//    uint16_t index_part1;
//    uint16_t dimension;
//};
//
//LUISA_DEVICE_CALLABLE constexpr int2 extended_gcd(uint32_t a, uint32_t b) noexcept {
//    if (b == 0) { return {1, 0}; }
//    auto d = static_cast<int32_t>(a / b);
//    auto p = extended_gcd(b, a % b);
//    return {p.y, p.x - d * p.y};
//}
//
//template<uint32_t base>
//LUISA_DEVICE_CALLABLE constexpr uint2 ceiling_scale_and_exponent(uint32_t x) noexcept {
//    auto exp = 0u;
//    auto scale = 1u;
//    while (scale < x) {
//        scale *= base;
//        exp++;
//    }
//    return {scale, exp};
//}
//
//LUISA_DEVICE_CALLABLE constexpr uint32_t multiplicative_inverse(int32_t a, int32_t n) noexcept {
//    auto gcd = extended_gcd(a, n);
//    auto m = gcd.x - (gcd.x / n) * n;
//    return static_cast<uint32_t>(m < 0 ? m + n : m);
//}
//
//LUISA_DEVICE_CALLABLE constexpr auto reverse_bits_u32(uint32_t n) noexcept {
//    n = (n << 16u) | (n >> 16u);
//    n = ((n & 0x00ff00ffu) << 8u) | ((n & 0xff00ff00u) >> 8u);
//    n = ((n & 0x0f0f0f0fu) << 4u) | ((n & 0xf0f0f0f0u) >> 4u);
//    n = ((n & 0x33333333u) << 2u) | ((n & 0xccccccccu) >> 2u);
//    n = ((n & 0x55555555u) << 1u) | ((n & 0xaaaaaaaau) >> 1u);
//    return n;
//}
//
//LUISA_DEVICE_CALLABLE constexpr auto reverse_bits_u64(uint64_t n) noexcept {
//    auto n0 = static_cast<uint64_t>(reverse_bits_u32(static_cast<uint32_t>(n)));
//    auto n1 = static_cast<uint64_t>(reverse_bits_u32(static_cast<uint32_t>((n >> 32u))));
//    return (n0 << 32u) | n1;
//}
//
//LUISA_DEVICE_CALLABLE constexpr float radical_inverse_base2(uint64_t a) noexcept {
//    return reverse_bits_u64(a) * 5.4210108624275222e-20f;
//}
//
//LUISA_DEVICE_CALLABLE constexpr float radical_inverse_base3(uint64_t a) noexcept {
//    constexpr auto base = 3u;
//    constexpr auto inv_base = 1.0f / base;
//    auto reversed_digits = 0ull;
//    auto inv_base_n = 1.0f;
//    while (a != 0) {
//        auto next = a / base;
//        auto digit = a - next * base;
//        reversed_digits = reversed_digits * base + digit;
//        inv_base_n *= inv_base;
//        a = next;
//    }
//    auto r = reversed_digits * inv_base_n;
//    return r < ONE_MINUS_EPSILON ? r : ONE_MINUS_EPSILON;
//}
//
//template<uint32_t base>
//LUISA_DEVICE_CALLABLE constexpr uint64_t inverse_radical_inverse(uint64_t inverse, uint32_t n_digits) noexcept {
//    auto index = 0ull;
//    for (auto i = 0u; i < n_digits; ++i) {
//        auto digit = inverse % base;
//        inverse /= base;
//        index = index * base + digit;
//    }
//    return index;
//}
//
//LUISA_DEVICE_CALLABLE static float scrambled_radical_inverse(uint32_t base, uint64_t a, const uint16_t *perm) noexcept {
//    auto inv_base = 1.0f / base;
//    auto reversed_digits = 0ull;
//    auto inv_base_n = 1.0f;
//    while (a != 0) {
//        auto next = a / base;
//        auto digit = a - next * base;
//        reversed_digits = reversed_digits * base + perm[digit];
//        inv_base_n *= inv_base;
//        a = next;
//    }
//    return fminf(inv_base_n * (reversed_digits + inv_base * perm[0] / (1.0f - inv_base)), ONE_MINUS_EPSILON);
//}
//
//LUISA_DEVICE_CALLABLE inline HaltonSamplerState halton_sampler_make_state(uint2 pixel, uint64_t sample_num, const HaltonSamplerParameters &param) noexcept {
//    auto pmx = pixel.x % HALTON_SAMPLER_PATCH_SIZE;
//    auto pmy = pixel.y % HALTON_SAMPLER_PATCH_SIZE;
//    auto base2_dim_offset = static_cast<uint64_t>(inverse_radical_inverse<2>(pmx, param.base2_exponent));
//    auto base3_dim_offset = static_cast<uint64_t>(inverse_radical_inverse<3>(pmy, param.base3_exponent));
//    auto base2_offset = (base2_dim_offset * param.base3_scale * param.multiplicative_inverses[0]);
//    auto base3_offset = (base3_dim_offset * param.base2_scale * param.multiplicative_inverses[1]);
//    auto initial_index = (base2_offset + base3_offset) % param.sample_stride;
//    auto index = initial_index + sample_num * param.sample_stride;
//    auto index_part0 = static_cast<uint32_t>(index >> 16u);
//    auto index_part1 = static_cast<uint16_t>(index);
//    return {index_part0, index_part1, 0u};
//}
//
//LUISA_DEVICE_CALLABLE inline float halton_sampler_generate_sample(HaltonSamplerState &state, const HaltonSamplerParameters &param) noexcept {
//    auto index = (static_cast<uint64_t>(state.index_part0) << 16u) | state.index_part1;
//    auto dim = state.dimension++;
//    switch (dim) {
//        case 0:
//            return radical_inverse_base2(index >> param.base2_exponent);
//        case 1:
//            return radical_inverse_base3(index / param.base3_scale);
//        default:
//            return scrambled_radical_inverse(param.primes[dim], index, &param.digit_permutations[param.prime_sums[dim]]);
//    }
//}
//
//}
//
//#ifndef LUISA_DEVICE_COMPATIBLE
//
//#include <vector>
//#include <random>
//
//namespace gr {
//
//class HaltonSampler {
//
//private:
//    static std::vector<uint16_t> _digit_permutations;
//
//private:
//    uint32_t _width;
//    uint32_t _height;
//    uint32_t _base2_scale;
//    uint32_t _base3_scale;
//    uint32_t _base2_exponent;
//    uint32_t _base3_exponent;
//    uint32_t _sample_stride;
//    uint32_t _multiplicative_inverses[2]{};
//
//public:
//    HaltonSampler(uint32_t width, uint32_t height) noexcept;
//    void update_parameters(HaltonSamplerParameters &param) const noexcept;
//
//};
//
//}
//
//#endif  // !defined DEVICE_COMPATIBLE