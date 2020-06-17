//
// Created by Mike Smith on 2020/6/17.
//

#pragma once

#include <memory>
#include "type_desc.h"

namespace luisa {

enum struct ScalarType {
    BYTE, UBYTE,
    SHORT, USHORT,
    INT, UINT,
    LONG, ULONG,
    BOOL,
    FLOAT
};

struct ScalarDesc : public TypeDesc {
    ScalarType type;
    explicit ScalarDesc(ScalarType type) noexcept: type{type} {}
    void accept(const TypeDescVisitor &compiler) const override { compiler._visit(*this); }
};

namespace detail {

template<>
struct TypeDescCreator<int32_t> {
    static std::unique_ptr<TypeDesc> create() noexcept {
        return std::make_unique<ScalarDesc>(ScalarType::INT);
    }
};

template<>
struct TypeDescCreator<uint32_t> {
    static std::unique_ptr<TypeDesc> create() noexcept {
        return std::make_unique<ScalarDesc>(ScalarType::UINT);
    }
};

template<>
struct TypeDescCreator<float> {
    static std::unique_ptr<TypeDesc> create() noexcept {
        return std::make_unique<ScalarDesc>(ScalarType::FLOAT);
    }
};

}

}
