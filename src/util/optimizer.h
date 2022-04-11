//
// Created by ChenXin on 2022/4/11.
//

#pragma once

namespace luisa::render {

enum class Optimizer {
    BGD = 0,
    SGD = 1,
    AdaGrad = 2,
    Adam = 3,
    ATN = 4,
    LDGD = 5,
};

}// namespace luisa::render