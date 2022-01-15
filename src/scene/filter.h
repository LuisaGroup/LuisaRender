//
// Created by Mike on 2021/12/8.
//

#pragma once

#include <array>

#include <scene/scene_node.h>
#include <scene/sampler.h>

namespace luisa::render {

class Filter : public SceneNode {

public:
    static constexpr auto look_up_table_size = 63u;

public:
    struct Sample {
        Float2 offset;
        Float weight;
    };

    class Instance {

    private:
        const Filter *_filter;
        std::array<float, look_up_table_size> _lut{};
        std::array<float, look_up_table_size - 1u> _pdf{};
        std::array<float, look_up_table_size - 1u> _alias_probs{};
        std::array<uint, look_up_table_size - 1u> _alias_indices{};

    public:
        explicit Instance(const Filter *filter) noexcept;
        virtual ~Instance() noexcept = default;
        [[nodiscard]] auto node() const noexcept { return _filter; }
        [[nodiscard]] auto look_up_table() const noexcept { return luisa::span{_lut}; }
        [[nodiscard]] auto pdf_table() const noexcept { return luisa::span{_pdf}; }
        [[nodiscard]] auto alias_table_indices() const noexcept { return luisa::span{_alias_indices}; }
        [[nodiscard]] auto alias_table_probabilities() const noexcept { return luisa::span{_alias_probs}; }
        [[nodiscard]] virtual Sample sample(Sampler::Instance &sampler) const noexcept;
    };

private:
    float2 _radius;

public:
    Filter(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] auto radius() const noexcept { return _radius; }
    [[nodiscard]] virtual float evaluate(float x) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept;
};

}
