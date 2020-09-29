//
// Created by Mike Smith on 2020/9/5.
//

#pragma once

#include <optional>

#include <compute/buffer.h>
#include <compute/device.h>

namespace luisa::render {

using compute::Device;
using compute::BufferView;
using compute::Expr;

struct ShaderSelection {
    Expr<uint> type;
    Expr<uint> index;
    Expr<float> prob;
    Expr<float> weight;
};

struct Interaction {
    
    enum Component {
        COMPONENT_MISS = 1u << 0u,
        COMPONENT_PI = 1u << 1u,
        COMPONENT_WO = 1u << 2u,
        COMPONENT_NG = 1u << 3u,
        COMPONENT_NS = 1u << 4u,
        COMPONENT_UV = 1u << 5u,
        COMPONENT_DISTANCE = 1u << 6u,
        COMPONENT_SHADER = 1u << 7u,
        COMPONENT_PDF = 1u << 8u,
        
        COMPONENT_NONE = 0u,
        COMPONENT_ALL = 0xffffffffu
    };
    
    std::optional<Expr<bool>> miss;
    std::optional<Expr<float3>> pi;
    std::optional<Expr<float3>> wo;
    std::optional<Expr<float>> distance;
    std::optional<Expr<float3>> ng;
    std::optional<Expr<float3>> ns;
    std::optional<Expr<float2>> uv;
    std::optional<Expr<float>> pdf;
    std::optional<ShaderSelection> shader;
};

class InteractionBuffers {

public:
    struct ShaderBuffers {
        BufferView<uint> type;
        BufferView<uint> index;
        BufferView<float> prob;
        BufferView<float> weight;
    };

private:
    size_t _size{0u};
    uint32_t _components{Interaction::COMPONENT_NONE};

public:
    BufferView<bool> miss;
    BufferView<float3> pi;
    BufferView<float3> wo;
    BufferView<float> distance;
    BufferView<float3> ng;
    BufferView<float3> ns;
    BufferView<float2> uv;
    BufferView<float> pdf;
    ShaderBuffers shader;

public:
    [[nodiscard]] static InteractionBuffers create(Device *device, size_t size, uint32_t components = Interaction::COMPONENT_ALL) noexcept {
        InteractionBuffers buffers;
        buffers._size = size;
        buffers._components = components;
        if (buffers.has_miss()) { buffers.miss = device->allocate_buffer<bool>(size); }
        if (buffers.has_pi()) { buffers.pi = device->allocate_buffer<float3>(size); }
        if (buffers.has_wo()) { buffers.wo = device->allocate_buffer<float3>(size); }
        if (buffers.has_distance()) { buffers.distance = device->allocate_buffer<float>(size); }
        if (buffers.has_ng()) { buffers.ng = device->allocate_buffer<float3>(size); }
        if (buffers.has_ns()) { buffers.ns = device->allocate_buffer<float3>(size); }
        if (buffers.has_uv()) { buffers.uv = device->allocate_buffer<float2>(size); }
        if (buffers.has_pdf()) { buffers.pdf = device->allocate_buffer<float>(size); }
        if (buffers.has_shader()) {
            buffers.shader.type = device->allocate_buffer<uint>(size);
            buffers.shader.index = device->allocate_buffer<uint>(size);
            buffers.shader.prob = device->allocate_buffer<float>(size);
            buffers.shader.weight = device->allocate_buffer<float>(size);
        }
        return buffers;
    }
    
    [[nodiscard]] bool has_miss() const noexcept { return _components & Interaction::COMPONENT_MISS; }
    [[nodiscard]] bool has_pi() const noexcept { return _components & Interaction::COMPONENT_PI; }
    [[nodiscard]] bool has_wo() const noexcept { return _components & Interaction::COMPONENT_WO; }
    [[nodiscard]] bool has_distance() const noexcept { return _components & Interaction::COMPONENT_DISTANCE; }
    [[nodiscard]] bool has_ng() const noexcept { return _components & Interaction::COMPONENT_NG; }
    [[nodiscard]] bool has_ns() const noexcept { return _components & Interaction::COMPONENT_NS; }
    [[nodiscard]] bool has_uv() const noexcept { return _components & Interaction::COMPONENT_UV; }
    [[nodiscard]] bool has_pdf() const noexcept { return _components & Interaction::COMPONENT_PDF; }
    [[nodiscard]] bool has_shader() const noexcept { return _components & Interaction::COMPONENT_SHADER; }
    
    [[nodiscard]] uint flags() const noexcept { return _components; }
    [[nodiscard]] size_t size() const noexcept { return _size; }
    
    template<typename Index>
    void emplace(Index &&index, const Interaction &interaction) {
        using namespace luisa::compute;
        using namespace luisa::compute::dsl;
        if (has_miss() && interaction.miss.has_value()) { miss[std::forward<Index>(index)] = *interaction.miss; }
        if (has_pi() && interaction.pi.has_value()) { pi[std::forward<Index>(index)] = *interaction.pi; }
        if (has_wo() && interaction.wo.has_value()) { wo[std::forward<Index>(index)] = *interaction.wo; }
        if (has_distance() && interaction.distance.has_value()) { distance[std::forward<Index>(index)] = *interaction.distance; }
        if (has_ng() && interaction.ng.has_value()) { ng[std::forward<Index>(index)] = *interaction.ng; }
        if (has_ns() && interaction.ns.has_value()) { ns[std::forward<Index>(index)] = *interaction.ns; }
        if (has_uv() && interaction.uv.has_value()) { uv[std::forward<Index>(index)] = *interaction.uv; }
        if (has_pdf() && interaction.pdf.has_value()) { pdf[std::forward<Index>(index)] = *interaction.pdf; }
        if (has_shader() && interaction.shader.has_value()) { shader[std::forward<Index>(index)] = *interaction.shader; }
    }
};

}
