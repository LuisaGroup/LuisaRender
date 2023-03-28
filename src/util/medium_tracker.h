//
// Created by Mike Smith on 2022/2/19.
//

#pragma once

#include <dsl/syntax.h>
#include <base/spectrum.h>
#include <base/medium.h>

namespace luisa::render {

#define TEST_COND all(dispatch_id().xy() == make_uint2(395, 853))
//#define TEST_COND false

using compute::ArrayVar;
using compute::Bool;
using compute::Expr;
using compute::Float4;
using compute::UInt;
using compute::Var;
using compute::Printer;

struct MediumInfo {
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
    Printer &_printer;

public:
    explicit MediumTracker(Printer &printer) noexcept;

protected:
    [[nodiscard]] auto &printer() noexcept { return _printer; }
    [[nodiscard]] const auto &printer() const noexcept { return _printer; }

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

LUISA_STRUCT(luisa::render::MediumInfo, medium_tag){};
