//
// Created by Mike Smith on 2022/2/19.
//

#pragma once

#include <luisa-compute.h>
#include <dsl/syntax.h>
#include <base/spectrum.h>
#include <base/medium.h>

namespace luisa::render {

using compute::ArrayVar;
using compute::Bool;
using compute::Expr;
using compute::Float4;
using compute::UInt;
using compute::Var;

struct MediumInfo {
    uint priority{0u};
    uint medium_tag{Medium::INVALID_TAG};
};

[[nodiscard]] Bool equal(Expr<MediumInfo> a, Expr<MediumInfo> b) noexcept;

class MediumTracker {

public:
    static constexpr auto capacity = 32u;

private:
    ArrayVar<uint, capacity> _priority_list;
    ArrayVar<MediumInfo, capacity> _medium_list;
    UInt _size;

public:
    explicit MediumTracker() noexcept;
    MediumTracker(const MediumTracker &) = default;

public:
    [[nodiscard]] Var<MediumInfo> current() const noexcept;
    [[nodiscard]] Bool vacuum() const noexcept;
    [[nodiscard]] Bool true_hit(Expr<uint> priority) const noexcept;
    void enter(Expr<uint> priority, Expr<MediumInfo> value) noexcept;
    void exit(Expr<uint> priority, Expr<MediumInfo> value) noexcept;
    [[nodiscard]] Bool exist(Expr<uint> priority, Expr<MediumInfo> value) noexcept;
    [[nodiscard]] UInt size() const noexcept { return _size; }
};

}// namespace luisa::render

LUISA_STRUCT(
    luisa::render::MediumInfo,
    priority,
    medium_tag) {

    [[nodiscard]] luisa::compute::Bool equal(luisa::compute::Expr<luisa::render::MediumInfo> v) const noexcept {
        return (this->medium_tag == v.medium_tag) &
               (this->priority == v.priority);
    }
};

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::MediumTracker)

[[nodiscard]] inline luisa::compute::Var<luisa::render::MediumInfo> make_medium_info(
    luisa::compute::UInt priority, luisa::compute::UInt medium_tag) noexcept {
    return luisa::compute::def<luisa::render::MediumInfo>(priority, medium_tag);
}
