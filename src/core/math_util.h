//
// Created by Mike Smith on 2020/1/30.
//

#pragma once

#include <algorithm>
#include <cmath>

#include "data_types.h"

namespace luisa {

inline namespace math {

inline namespace constants {

constexpr auto pi = 3.14159265358979323846264338327950288f;
constexpr auto pi_over_two = 1.57079632679489661923132169163975144f;
constexpr auto pi_over_four = 0.785398163397448309615660845819875721f;
constexpr auto inv_pi = 0.318309886183790671537767526745028724f;
constexpr auto two_over_pi = 0.636619772367581343075535053490057448f;
constexpr auto sqrt_two = 1.41421356237309504880168872420969808f;
constexpr auto inv_sqrt_two = 0.707106781186547524400844362104849039f;

constexpr auto prime_number_count = 512u;

constexpr uint prime_numbers [[maybe_unused]][prime_number_count] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
    89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
    181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277,
    281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
    397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
    503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617,
    619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739,
    743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859,
    863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991,
    997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
    1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201,
    1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301,
    1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433,
    1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531,
    1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621,
    1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747,
    1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873,
    1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997,
    1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099,
    2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237,
    2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341,
    2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441,
    2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591,
    2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693,
    2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797,
    2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917,
    2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049,
    3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191,
    3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319,
    3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449,
    3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547,
    3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671};

constexpr uint prime_number_prefix_sums [[maybe_unused]][prime_number_count] = {
    0, 2, 5, 10, 17, 28, 41, 58, 77, 100, 129, 160, 197, 238, 281, 328, 381, 440, 501, 568,
    639, 712, 791, 874, 963, 1060, 1161, 1264, 1371, 1480, 1593, 1720, 1851, 1988, 2127, 2276,
    2427, 2584, 2747, 2914, 3087, 3266, 3447, 3638, 3831, 4028, 4227, 4438, 4661, 4888, 5117,
    5350, 5589, 5830, 6081, 6338, 6601, 6870, 7141, 7418, 7699, 7982, 8275, 8582, 8893, 9206,
    9523, 9854, 10191, 10538, 10887, 11240, 11599, 11966, 12339, 12718, 13101, 13490, 13887,
    14288, 14697, 15116, 15537, 15968, 16401, 16840, 17283, 17732, 18189, 18650, 19113, 19580,
    20059, 20546, 21037, 21536, 22039, 22548, 23069, 23592, 24133, 24680, 25237, 25800, 26369,
    26940, 27517, 28104, 28697, 29296, 29897, 30504, 31117, 31734, 32353, 32984, 33625, 34268,
    34915, 35568, 36227, 36888, 37561, 38238, 38921, 39612, 40313, 41022, 41741, 42468, 43201,
    43940, 44683, 45434, 46191, 46952, 47721, 48494, 49281, 50078, 50887, 51698, 52519, 53342,
    54169, 54998, 55837, 56690, 57547, 58406, 59269, 60146, 61027, 61910, 62797, 63704, 64615,
    65534, 66463, 67400, 68341, 69288, 70241, 71208, 72179, 73156, 74139, 75130, 76127, 77136,
    78149, 79168, 80189, 81220, 82253, 83292, 84341, 85392, 86453, 87516, 88585, 89672, 90763,
    91856, 92953, 94056, 95165, 96282, 97405, 98534, 99685, 100838, 102001, 103172, 104353,
    105540, 106733, 107934, 109147, 110364, 111587, 112816, 114047, 115284, 116533, 117792,
    119069, 120348, 121631, 122920, 124211, 125508, 126809, 128112, 129419, 130738, 132059,
    133386, 134747, 136114, 137487, 138868, 140267, 141676, 143099, 144526, 145955, 147388,
    148827, 150274, 151725, 153178, 154637, 156108, 157589, 159072, 160559, 162048, 163541,
    165040, 166551, 168074, 169605, 171148, 172697, 174250, 175809, 177376, 178947, 180526,
    182109, 183706, 185307, 186914, 188523, 190136, 191755, 193376, 195003, 196640, 198297,
    199960, 201627, 203296, 204989, 206686, 208385, 210094, 211815, 213538, 215271, 217012,
    218759, 220512, 222271, 224048, 225831, 227618, 229407, 231208, 233019, 234842, 236673,
    238520, 240381, 242248, 244119, 245992, 247869, 249748, 251637, 253538, 255445, 257358,
    259289, 261222, 263171, 265122, 267095, 269074, 271061, 273054, 275051, 277050, 279053,
    281064, 283081, 285108, 287137, 289176, 291229, 293292, 295361, 297442, 299525, 301612,
    303701, 305800, 307911, 310024, 312153, 314284, 316421, 318562, 320705, 322858, 325019,
    327198, 329401, 331608, 333821, 336042, 338279, 340518, 342761, 345012, 347279, 349548,
    351821, 354102, 356389, 358682, 360979, 363288, 365599, 367932, 370271, 372612, 374959,
    377310, 379667, 382038, 384415, 386796, 389179, 391568, 393961, 396360, 398771, 401188,
    403611, 406048, 408489, 410936, 413395, 415862, 418335, 420812, 423315, 425836, 428367,
    430906, 433449, 435998, 438549, 441106, 443685, 446276, 448869, 451478, 454095, 456716,
    459349, 461996, 464653, 467312, 469975, 472646, 475323, 478006, 480693, 483382, 486075,
    488774, 491481, 494192, 496905, 499624, 502353, 505084, 507825, 510574, 513327, 516094,
    518871, 521660, 524451, 527248, 530049, 532852, 535671, 538504, 541341, 544184, 547035,
    549892, 552753, 555632, 558519, 561416, 564319, 567228, 570145, 573072, 576011, 578964,
    581921, 584884, 587853, 590824, 593823, 596824, 599835, 602854, 605877, 608914, 611955,
    615004, 618065, 621132, 624211, 627294, 630383, 633492, 636611, 639732, 642869, 646032,
    649199, 652368, 655549, 658736, 661927, 665130, 668339, 671556, 674777, 678006, 681257,
    684510, 687767, 691026, 694297, 697596, 700897, 704204, 707517, 710836, 714159, 717488,
    720819, 724162, 727509, 730868, 734229, 737600, 740973, 744362, 747753, 751160, 754573,
    758006, 761455, 764912, 768373, 771836, 775303, 778772, 782263, 785762, 789273, 792790,
    796317, 799846, 803379, 806918, 810459, 814006, 817563, 821122, 824693, 828274, 831857,
    835450, 839057, 842670, 846287, 849910, 853541, 857178, 860821, 864480};
    
}// namespace constants

// bit manipulation function
[[nodiscard]] constexpr auto next_pow_of_two(uint v) noexcept {
    v--;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    v++;
    return v;
}

// Scalar Functions
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::sin;
using std::tan;

using std::sqrt;

using std::ceil;
using std::floor;
using std::round;

using std::exp;
using std::log;
using std::log10;
using std::log2;
using std::pow;

using std::max;
using std::min;

using std::abs;

[[nodiscard]] constexpr float radians(float deg) noexcept { return deg * constants::pi / 180.0f; }
[[nodiscard]] constexpr float degrees(float rad) noexcept { return rad * constants::inv_pi * 180.0f; }

#define MAKE_VECTOR_UNARY_FUNC(func)                                          \
    template<typename T, uint N>                                              \
    [[nodiscard]] constexpr auto func(Vector<T, N> v) noexcept {              \
        static_assert(N == 2 || N == 3 || N == 4);                            \
        if constexpr (N == 2) {                                               \
            return Vector<T, 2>{func(v.x), func(v.y)};                        \
        } else if constexpr (N == 3) {                                        \
            return Vector<T, 3>(func(v.x), func(v.y), func(v.z));             \
        } else {                                                              \
            return Vector<T, 4>(func(v.x), func(v.y), func(v.z), func(v.w));  \
        }                                                                     \
    }

MAKE_VECTOR_UNARY_FUNC(acos)
MAKE_VECTOR_UNARY_FUNC(asin)
MAKE_VECTOR_UNARY_FUNC(atan)
MAKE_VECTOR_UNARY_FUNC(cos)
MAKE_VECTOR_UNARY_FUNC(sin)
MAKE_VECTOR_UNARY_FUNC(tan)
MAKE_VECTOR_UNARY_FUNC(sqrt)
MAKE_VECTOR_UNARY_FUNC(ceil)
MAKE_VECTOR_UNARY_FUNC(floor)
MAKE_VECTOR_UNARY_FUNC(round)
MAKE_VECTOR_UNARY_FUNC(exp)
MAKE_VECTOR_UNARY_FUNC(log)
MAKE_VECTOR_UNARY_FUNC(log10)
MAKE_VECTOR_UNARY_FUNC(log2)
MAKE_VECTOR_UNARY_FUNC(abs)
MAKE_VECTOR_UNARY_FUNC(radians)
MAKE_VECTOR_UNARY_FUNC(degrees)

#undef MAKE_VECTOR_UNARY_FUNC

#define MAKE_VECTOR_BINARY_FUNC(func)                                                             \
    template<typename T, uint N>                                                                  \
    [[nodiscard]] constexpr auto func(Vector<T, N> v, Vector<T, N> u) noexcept {                  \
        static_assert(N == 2 || N == 3 || N == 4);                                                \
        if constexpr (N == 2) {                                                                   \
            return Vector<T, 2>{func(v.x, u.x), func(v.y, u.y)};                                  \
        } else if constexpr (N == 3) {                                                            \
            return Vector<T, 3>(func(v.x, u.x), func(v.y, u.y), func(v.z, u.z));                  \
        } else {                                                                                  \
            return Vector<T, 4>(func(v.x, u.x), func(v.y, u.y), func(v.z, u.z), func(v.w, u.w));  \
        }                                                                                         \
    }                                                                                             \
    template<typename T, uint N>                                                                  \
    [[nodiscard]] constexpr auto func(T v, Vector<T, N> u) noexcept {                             \
        static_assert(N == 2 || N == 3 || N == 4);                                                \
        if constexpr (N == 2) {                                                                   \
            return Vector<T, 2>{func(v, u.x), func(v, u.y)};                                      \
        } else if constexpr (N == 3) {                                                            \
            return Vector<T, 3>(func(v, u.x), func(v, u.y), func(v, u.z));                        \
        } else {                                                                                  \
            return Vector<T, 4>(func(v, u.x), func(v, u.y), func(v, u.z), func(v, u.w));          \
        }                                                                                         \
    }                                                                                             \
    template<typename T, uint N>                                                                  \
    [[nodiscard]] constexpr auto func(Vector<T, N> v, T u) noexcept {                             \
        static_assert(N == 2 || N == 3 || N == 4);                                                \
        if constexpr (N == 2) {                                                                   \
            return Vector<T, 2>{func(v.x, u), func(v.y, u)};                                      \
        } else if constexpr (N == 3) {                                                            \
            return Vector<T, 3>(func(v.x, u), func(v.y, u), func(v.z, u));                        \
        } else {                                                                                  \
            return Vector<T, 4>(func(v.x, u), func(v.y, u), func(v.z, u), func(v.w, u));          \
        }                                                                                         \
    }

MAKE_VECTOR_BINARY_FUNC(atan2)
MAKE_VECTOR_BINARY_FUNC(pow)
MAKE_VECTOR_BINARY_FUNC(min)
MAKE_VECTOR_BINARY_FUNC(max)

#undef MAKE_VECTOR_BINARY_FUNC

template<typename T, typename F>
[[nodiscard]] constexpr auto select(bool pred, T t, F f) noexcept {
    return pred ? t : f;
}

template<typename T, uint N, std::enable_if_t<scalar::is_scalar<T>, int> = 0>
[[nodiscard]] constexpr auto select(Vector<bool, N> pred, Vector<T, N> t, Vector<T, N> f) noexcept {
    static_assert(N == 2 || N == 3 || N == 4);
    if constexpr (N == 2) {
        return Vector<T, N>{select(pred.x, t.x, f.x), select(pred.y, t.y, f.y)};
    } else if constexpr (N == 3) {
        return Vector<T, N>{select(pred.x, t.x, f.x), select(pred.y, t.y, f.y), select(pred.z, t.z, f.z)};
    } else {
        return Vector<T, N>{select(pred.x, t.x, f.x), select(pred.y, t.y, f.y), select(pred.z, t.z, f.z), select(pred.w, t.w, f.w)};
    }
}

template<typename A, typename B>
[[nodiscard]] constexpr auto lerp(A a, B b, float t) noexcept {
    return a + t * (b - a);
}

template<typename X, typename A, typename B>
[[nodiscard]] constexpr auto clamp(X x, A a, B b) noexcept {
    return min(max(x, a), b);
}

// Vector Functions
template<uint N>
[[nodiscard]] constexpr auto dot(Vector<float, N> u, Vector<float, N> v) noexcept {
    static_assert(N == 2 || N == 3 || N == 4);
    if constexpr (N == 2) {
        return u.x * v.x + u.y * v.y;
    } else if constexpr (N == 3) {
        return u.x * v.x + u.y * v.y + u.z * v.z;
    } else {
        return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
    }
}

template<uint N>
[[nodiscard]] constexpr auto length(Vector<float, N> u) noexcept {
    return sqrt(dot(u, u));
}

template<uint N>
[[nodiscard]] constexpr auto normalize(Vector<float, N> u) noexcept {
    return u * (1.0f / length(u));
}

template<uint N>
[[nodiscard]] constexpr auto distance(Vector<float, N> u, Vector<float, N> v) noexcept {
    return length(u - v);
}

[[nodiscard]] constexpr auto cross(float3 u, float3 v) noexcept {
    return make_float3(u.y * v.z - v.y * u.z,
                       u.z * v.x - v.z * u.x,
                       u.x * v.y - v.x * u.y);
}

// Matrix Functions
[[nodiscard]] constexpr auto transpose(const float3x3 m) noexcept {
    return make_float3x3(m[0].x, m[1].x, m[2].x,
                         m[0].y, m[1].y, m[2].y,
                         m[0].z, m[1].z, m[2].z);
}

[[nodiscard]] constexpr auto transpose(const float4x4 m) noexcept {
    return make_float4x4(m[0].x, m[1].x, m[2].x, m[3].x,
                         m[0].y, m[1].y, m[2].y, m[3].y,
                         m[0].z, m[1].z, m[2].z, m[3].z,
                         m[0].w, m[1].w, m[2].w, m[3].w);
}

[[nodiscard]] constexpr auto inverse(const float3x3 m) noexcept {// from GLM
    const auto one_over_determinant = 1.0f /
                                      (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) - m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) + m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
    return make_float3x3(
        (m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,
        (m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,
        (m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,
        (m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,
        (m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,
        (m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,
        (m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,
        (m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,
        (m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant);
}

[[nodiscard]] constexpr auto inverse(const float4x4 m) noexcept {// from GLM
    const auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
    const auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
    const auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
    const auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
    const auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
    const auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
    const auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
    const auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
    const auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
    const auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
    const auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
    const auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
    const auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
    const auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
    const auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
    const auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
    const auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
    const auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
    const auto fac0 = make_float4(coef00, coef00, coef02, coef03);
    const auto fac1 = make_float4(coef04, coef04, coef06, coef07);
    const auto fac2 = make_float4(coef08, coef08, coef10, coef11);
    const auto fac3 = make_float4(coef12, coef12, coef14, coef15);
    const auto fac4 = make_float4(coef16, coef16, coef18, coef19);
    const auto fac5 = make_float4(coef20, coef20, coef22, coef23);
    const auto Vec0 = make_float4(m[1].x, m[0].x, m[0].x, m[0].x);
    const auto Vec1 = make_float4(m[1].y, m[0].y, m[0].y, m[0].y);
    const auto Vec2 = make_float4(m[1].z, m[0].z, m[0].z, m[0].z);
    const auto Vec3 = make_float4(m[1].w, m[0].w, m[0].w, m[0].w);
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    const auto sign_a = make_float4(+1, -1, +1, -1);
    const auto sign_b = make_float4(-1, +1, -1, +1);
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m[0] * make_float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
    const auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
    const auto one_over_determinant = 1.0f / dot1;
    return make_float4x4(inv_0 * one_over_determinant,
                         inv_1 * one_over_determinant,
                         inv_2 * one_over_determinant,
                         inv_3 * one_over_determinant);
}

// transforms
constexpr float4x4 translation(const float3 v) noexcept {
    return make_float4x4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        v.x, v.y, v.z, 1.0f);
}

inline float4x4 rotation(const float3 axis, float angle) noexcept {
    
    auto c = cos(angle);
    auto s = sin(angle);
    auto a = normalize(axis);
    auto t = (1.0f - c) * a;
    
    return make_float4x4(
        c + t.x * a.x, t.x * a.y + s * a.z, t.x * a.z - s * a.y, 0.0f,
        t.y * a.x - s * a.z, c + t.y * a.y, t.y * a.z + s * a.x, 0.0f,
        t.z * a.x + s * a.y, t.z * a.y - s * a.x, c + t.z * a.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

constexpr float4x4 scaling(const float3 s) noexcept {
    return make_float4x4(
        s.x, 0.0f, 0.0f, 0.0f,
        0.0f, s.y, 0.0f, 0.0f,
        0.0f, 0.0f, s.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

}
}// namespace luisa::math
