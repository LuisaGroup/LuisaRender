//
// Created by Mike Smith on 2020/7/10.
//

#include "variable.h"

namespace luisa {

Variable::Variable(Function *function, uint32_t id) noexcept
    : _function{function}, _id{id} {}

}
