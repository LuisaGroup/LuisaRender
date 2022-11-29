//
// Created by Mike Smith on 2022/1/27.
//

#pragma once

#include <fstream>
#include <filesystem>

#include <core/stl.h>

namespace luisa::render {

// TODO
class IESProfile {

private:
    luisa::vector<float> _vertical_angles;
    luisa::vector<float> _horizontal_angles;
    luisa::vector<float> _candela_values;

private:
    IESProfile(luisa::vector<float> v_angles,
               luisa::vector<float> h_angles,
               luisa::vector<float> values) noexcept;

public:
    [[nodiscard]] static IESProfile parse(const std::filesystem::path &path) noexcept;
    [[nodiscard]] auto vertical_angles() const noexcept { return luisa::span{_vertical_angles}; }
    [[nodiscard]] auto horizontal_angles() const noexcept { return luisa::span{_horizontal_angles}; }
    [[nodiscard]] auto candela_values() const noexcept { return luisa::span{_candela_values}; }
};

}// namespace luisa::render
