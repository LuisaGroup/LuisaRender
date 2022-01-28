//
// Created by Mike Smith on 2022/1/27.
//

#include <core/logging.h>
#include <util/ies.h>

namespace luisa::render {

IESProfile IESProfile::parse(const std::filesystem::path &path) noexcept {
    std::ifstream file{path};
    if (!file) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to open IES profile '{}'.",
            path.string());
    }
    luisa::string line;
    line.reserve(1024u);
    std::getline(file, line);
    if (!line.starts_with("IESNA:")) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid IES profile '{}' "
            "with first line: {}.",
            path.string(), line);
    }
    while (!line.starts_with("TILT")) {
        std::getline(file, line);
    }
    if (line.starts_with("TILE=INCLUDE")) {
        std::getline(file, line);// <lamp to luminaire geometry>
        std::getline(file, line);// <number of tilt angles>
        std::getline(file, line);// <angles>
        std::getline(file, line);// <multiplying factors>
    }
    [[maybe_unused]] auto number_of_lamps = 0;
    [[maybe_unused]] auto lumens_per_lamp = 0;
    auto candela_multiplier = 0.0f;
    auto number_of_vertical_angles = 0u;
    auto number_of_horizontal_angles = 0u;
    file >>
        number_of_lamps >>
        lumens_per_lamp >>
        candela_multiplier >>
        number_of_vertical_angles >>
        number_of_horizontal_angles;
    std::getline(file, line);// <photometric type> <units type> <width> <length> <height>
    std::getline(file, line);// <ballast factor> <future use> <input watts>
    luisa::vector<float> vertical_angles;
    vertical_angles.reserve(number_of_vertical_angles);
    for (auto i = 0u; i < number_of_vertical_angles; i++) {
        vertical_angles.emplace_back(0.0f);
        file >> vertical_angles.back();
    }
    luisa::vector<float> horizontal_angles;
    horizontal_angles.reserve(number_of_horizontal_angles);
    for (auto i = 0u; i < number_of_horizontal_angles; i++) {
        horizontal_angles.emplace_back(0.0f);
        file >> horizontal_angles.back();
    }
    auto n = number_of_vertical_angles *
             number_of_horizontal_angles;
    luisa::vector<float> candela_values;
    candela_values.reserve(n);
    for (auto h = 0u; h < n; h++) {
        auto value = 0.0f;
        file >> value;
        candela_values.emplace_back(
            value * candela_multiplier);
    }
    return {std::move(vertical_angles),
            std::move(horizontal_angles),
            std::move(candela_values)};
}

inline IESProfile::IESProfile(
    luisa::vector<float> v_angles,
    luisa::vector<float> h_angles,
    luisa::vector<float> values) noexcept
    : _vertical_angles{std::move(v_angles)},
      _horizontal_angles{std::move(h_angles)},
      _candela_values{std::move(values)} {}

}// namespace luisa::render
