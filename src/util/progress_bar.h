//
// Created by Mike Smith on 2022/4/7.
//

#pragma once

#include <chrono>

namespace luisa::render {

// credit: https://github.com/AirGuanZ/agz-utils/blob/master/include/agz-utils/console/pbar.h#L137
class ProgressBar {

public:
    using clock_type = std::chrono::steady_clock;
    static constexpr auto complete_char = '=';
    static constexpr auto heading_char = '>';
    static constexpr auto incomplete_char = ' ';

private:
    double _progress;
    uint _width;
    clock_type::time_point _start;

public:
    explicit ProgressBar(uint width = 50u) noexcept;
    void reset() noexcept;
    void update(double progress) noexcept;
    void done() noexcept;
};

}// namespace luisa::render
