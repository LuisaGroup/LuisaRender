#include <util/imageio.h>
#include <base/pipeline.h>
#include <base/integrator.h>
#include <base/scene.h>

namespace luisa::render {

class GroupIntegrator;

class GroupIntegratorInstance final : public Integrator::Instance {
    luisa::vector<luisa::unique_ptr<Integrator::Instance>> _integrators;

public:
    explicit GroupIntegratorInstance(
        const GroupIntegrator *integrator,
        Pipeline &pipeline, CommandBuffer &cb) noexcept;
    void render(Stream &stream) noexcept override;
};

class GroupIntegrator final : public Integrator {
    luisa::vector<const Integrator *> _integrators;

public:
    GroupIntegrator(Scene *scene, const SceneNodeDesc *desc) noexcept : Integrator{scene, desc} {
        auto children = desc->property_node_list_or_default("integrators");
        luisa::vector<const Integrator *> integrators(children.size());
        for (auto i = 0u; i < children.size(); i++) {
            integrators[i] = scene->load_integrator(children[i]);
        }
        _integrators = std::move(integrators);
    }

    [[nodiscard]] luisa::vector<const Integrator *> integrators() const noexcept { return _integrators; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Integrator::Instance> build(Pipeline &pipeline, CommandBuffer &cb) const noexcept override {
        return luisa::make_unique<GroupIntegratorInstance>(this, pipeline, cb);
    }
};

GroupIntegratorInstance::GroupIntegratorInstance(
    const GroupIntegrator *group, Pipeline &pipeline, CommandBuffer &cb) noexcept
    : Integrator::Instance{pipeline, cb, group} {
    luisa::vector<luisa::unique_ptr<Integrator::Instance>> instances(group->integrators().size());
    for (auto i = 0u; i < group->integrators().size(); i++) {
        instances[i] = std::move(group->integrators()[i]->build(pipeline, cb));
    }
    _integrators = std::move(instances);
}

void GroupIntegratorInstance::render(Stream &stream) noexcept {
    for (auto i = 0u; i < _integrators.size(); i++) {
        _integrators[i]->render(stream);
    }
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GroupIntegrator)
