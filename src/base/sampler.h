//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <dsl/syntax.h>
#include <base/scene.h>

namespace luisa::render {

using compute::Expr;
using compute::Float;
using compute::Float2;

class Sampler : public Scene::Node {

private:
    uint2 _resolution{};
    uint _sample_count{};

private:
    virtual void _on_set_resolution() noexcept {}
    virtual void _on_set_sample_count() noexcept {}

public:
    Sampler() noexcept : Scene::Node{Scene::Node::Tag::SAMPLER} {}
    Sampler &set_resolution(uint2 r) noexcept;
    Sampler &set_sample_count(uint spp) noexcept;
    [[nodiscard]] auto resolution() const noexcept { return _resolution; }
    [[nodiscard]] auto sample_count() const noexcept { return _sample_count; }
    virtual void start(Expr<uint2> pixel, Expr<uint> sample_index) noexcept = 0;
    virtual void finish() noexcept = 0;
    [[nodiscard]] virtual Float generate_1d() noexcept = 0;
    [[nodiscard]] virtual Float2 generate_2d() noexcept = 0;
};

}
