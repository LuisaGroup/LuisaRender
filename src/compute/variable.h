//
// Created by Mike Smith on 2020/7/10.
//

#pragma once

#include <compute/type_desc.h>

namespace luisa::dsl {

class Variable {

private:
    const TypeDesc *_type;
    uint32_t _uid;
    
    Variable(const TypeDesc *type, uint32_t uid) : _type{type}, _uid{uid} {}
};

}
