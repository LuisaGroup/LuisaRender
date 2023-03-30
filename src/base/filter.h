//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <array>

#include <base/scene_node.h>
#include <base/sampler.h>

namespace luisa::render {

class Filter : public SceneNode {

public:
    static constexpr auto look_up_table_size = 64u;

public:
    struct Sample {
        Float2 offset;
        Float weight;
    };

    class Instance {

    private:
        const Pipeline &_pipeline;
        const Filter *_filter;
        std::array<float, look_up_table_size> _lut{};
        std::array<float, look_up_table_size - 1u> _pdf{};
        std::array<float, look_up_table_size - 1u> _alias_probs{};
        std::array<uint, look_up_table_size - 1u> _alias_indices{};

    public:
        Instance(const Pipeline &pipeline, const Filter *filter) noexcept;
        virtual ~Instance() noexcept = default;

        template<typename T = Filter>
            requires std::is_base_of_v<Filter, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_filter); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] auto look_up_table() const noexcept { return luisa::span{_lut}; }
        [[nodiscard]] auto pdf_table() const noexcept { return luisa::span{_pdf}; }
        [[nodiscard]] auto alias_table_indices() const noexcept { return luisa::span{_alias_indices}; }
        [[nodiscard]] auto alias_table_probabilities() const noexcept { return luisa::span{_alias_probs}; }
        [[nodiscard]] virtual Sample sample(Expr<float2> u) const noexcept;
    };

private:
    float _radius;
    float2 _shift;

public:
    Filter(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto radius() const noexcept { return _radius; }
    [[nodiscard]] auto shift() const noexcept { return _shift; }
    [[nodiscard]] virtual float evaluate(float x) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept;
};

}// namespace luisa::render

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(luisa::render::Filter::Instance)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(::luisa::render::Filter::Sample)
