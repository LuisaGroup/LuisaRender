//
// Created by Mike Smith on 2022/2/19.
//

#pragma once

#include <dsl/syntax.h>
#include <base/spectrum.h>

namespace luisa::render {

using compute::ArrayVar;
using compute::Bool;
using compute::Expr;
using compute::Float4;
using compute::UInt;
using compute::Var;

struct MediumInfo {
    float eta;
};

class MediumTracker {

public:
    static constexpr auto capacity = 8u;

private:
    ArrayVar<uint, capacity> _priority_list;
    ArrayVar<MediumInfo, capacity> _medium_list;
    UInt _size;

public:
    MediumTracker() noexcept;
    [[nodiscard]] Var<MediumInfo> current() const noexcept;
    [[nodiscard]] Bool vacuum() const noexcept;
    [[nodiscard]] Bool true_hit(Expr<uint> priority) const noexcept;
    void enter(Expr<uint> priority, Expr<MediumInfo> value) noexcept;
    void exit(Expr<uint> priority) noexcept;
};

}// namespace luisa::render

LUISA_STRUCT(luisa::render::MediumInfo, eta){};
