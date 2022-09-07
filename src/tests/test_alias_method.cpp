//
// Created by Mike Smith on 2022/8/18.
//

#include <random>
#include <fstream>
#include <util/sampling.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

template<typename Table>
[[nodiscard]] auto sample_alias_table(
    const Table &table, uint n, float u_in) noexcept {
    using namespace luisa::compute;
    auto u = u_in * static_cast<float>(n);
    auto i = std::clamp(static_cast<uint>(u), 0u, n - 1u);
    auto u_remapped = fract(u);
    auto entry = table[i];
    return u_remapped < entry.prob ? i : entry.alias;
}

int main() {

    std::random_device random_device;
    std::mt19937 random{random_device()};
    std::uniform_real_distribution<float> dist{0.f, 1.f};

    constexpr auto value_count = 128u;
    constexpr auto sample_count = 1024ull * 1024ull * 1024ull;

    luisa::vector<float> values;
    values.reserve(value_count);

    std::ofstream file{"alias_data.json"};
    file << "{\n";

    for (auto i = 0u; i < value_count; ++i) {
        values.emplace_back(dist(random));
    }
    auto [alias_table, pdf_table] = create_alias_table(values);

    file << "  \"pdf\": [" << pdf_table[0];
    for (auto p : luisa::span{pdf_table}.subspan(1u)) {
        file << ", " << p;
    }
    file << "],\n";

    luisa::vector<uint64_t> bins(value_count, 0u);
    for (auto i = 0ull; i < sample_count; i++) {
        auto u = dist(random);
        auto index = sample_alias_table(alias_table, value_count, u);
        bins[index]++;
    }
    file << "  \"bins\": [" << bins[0];
    for (auto b : luisa::span{bins}.subspan(1u)) {
        file << ", " << b;
    }
    file << "],\n";

    auto freq = [sample_count](auto x) noexcept {
        return static_cast<float>(static_cast<double>(x) /
                                  static_cast<double>(sample_count));
    };
    file << "  \"frequencies\": [" << freq(bins[0]);
    for (auto b : luisa::span{bins}.subspan(1u)) {
        file << ", " << freq(b);
    }
    file << "],\n";

    file << "  \"error\": [" << freq(bins[0]) - pdf_table[0];
    for (auto i = 1u; i < value_count; i++) {
        file << ", " << freq(bins[i]) - pdf_table[i];
    }
    file << "],\n";
    file << "  \"relative_error\": [" << std::abs((freq(bins[0]) - pdf_table[0]) / pdf_table[0]);
    for (auto i = 1u; i < value_count; i++) {
        file << ", " << std::abs(freq(bins[i]) - pdf_table[i]) / pdf_table[i];
    }
    file << "]\n";
    file << "}\n";
}
