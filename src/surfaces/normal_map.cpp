//
// Created by Mike Smith on 2022/8/19.
//

#include <base/surface.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class NormalMap final : public Surface {

private:
    const Texture *_map;
    const Surface *_base;

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

public:
    NormalMap(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _map{scene->load_texture(desc->property_node_or_default("map"))},
          _base{scene->load_surface(desc->property_node_or_default("base"))} {
        LUISA_RENDER_CHECK_GENERIC_TEXTURE(NormalMap, map, 3);
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_null() const noexcept override { return _base == nullptr || _base->is_null(); }
};

class NormalMapInstance final : public Surface::Instance {

private:
    const Texture::Instance *_map;
    luisa::unique_ptr<Surface::Instance> _base;

public:
    NormalMapInstance(
        const Pipeline &pipeline, const NormalMap *surface,
        const Texture::Instance *map, luisa::unique_ptr<Surface::Instance> base) noexcept
        : Surface::Instance{pipeline, surface}, _map{map}, _base{std::move(base)} {}
    [[nodiscard]] auto map() const noexcept { return _map; }

private:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> NormalMap::_build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    LUISA_ASSERT(!is_null(), "Building null NormalMap.");
    luisa::unique_ptr<Surface::Instance> base;
    if (_base != nullptr && !_base->is_null()) [[likely]] {
        base = _base->build(pipeline, command_buffer);
    }
    auto map = pipeline.build_texture(command_buffer, _map);
    return luisa::make_unique<NormalMapInstance>(
        pipeline, this, map, std::move(base));
}

luisa::unique_ptr<Surface::Closure> NormalMapInstance::closure(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> eta_i, Expr<float> time) const noexcept {
    LUISA_ASSERT(_base != nullptr, "NormalMapInstance has no base surface.");
    if (_map == nullptr) { return _base->closure(it, swl, eta_i, time); }
    auto normal_local = 2.f * _map->evaluate(it, swl, time).xyz() - 1.f;
    auto normal = it.shading().local_to_world(normal_local);
    auto mapped_it = it;
    normal = ite(dot(normal, it.ng()) > 0.f, normal, it.shading().n());
    mapped_it.set_shading(Frame::make(normal, it.shading().u()));
    return _base->closure(mapped_it, swl, eta_i, time);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMap)
