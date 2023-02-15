//
// Created by ChenXin on 2023/2/13.
//

#include <base/medium.h>
#include <base/phase_function.h>
#include <base/texture.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

class HomogeneousMedium : public Medium {

private:
    const Texture *_sigma_a;                // absorption coefficient
    const Texture *_sigma_s;                // scattering coefficient
    const Texture *_Le;                     // emission coefficient
    const PhaseFunction *_phase_function;   // phase function

public:
    class HomogeneousMediumInstance;

    class HomogeneousMediumInstance : public Medium::Instance {

    private:
        const Texture::Instance *_sigma_a;
        const Texture::Instance *_sigma_s;
        const Texture::Instance *_Le;
        const PhaseFunction::Instance *_phase_function;


    public:
        [[nodiscard]] auto sigma_a() const noexcept { return _sigma_a; }
        [[nodiscard]] auto sigma_s() const noexcept { return _sigma_s; }
        [[nodiscard]] auto Le() const noexcept { return _Le; }
        [[nodiscard]] auto phase_function() const noexcept { return _phase_function; }

    protected:
        friend class HomogeneousMedium;

    public:
        HomogeneousMediumInstance(
            const Pipeline &pipeline, const Medium *medium,
            const Texture::Instance *sigma_a, const Texture::Instance *sigma_s,
            const Texture::Instance *Le, const PhaseFunction::Instance *phase_function) noexcept
            : Medium::Instance(pipeline, medium), _sigma_a{sigma_a}, _sigma_s{sigma_s}, _Le{Le}, _phase_function{phase_function} {}
    };

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        auto sigma_a = pipeline.build_texture(command_buffer, _sigma_a);
        auto sigma_s = pipeline.build_texture(command_buffer, _sigma_s);
        auto Le = pipeline.build_texture(command_buffer, _Le);
        auto phase_function = pipeline.build_phasefunction(command_buffer, _phase_function);
        return luisa::make_unique<HomogeneousMediumInstance>(pipeline, this, sigma_a, sigma_s, Le, phase_function);
    }

public:
    HomogeneousMedium(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Medium{scene, desc} {
        _sigma_a = scene->load_texture(desc->property_node_or_default("sigma_a"));
        _sigma_s = scene->load_texture(desc->property_node_or_default("sigma_s"));
        _Le = scene->load_texture(desc->property_node_or_default("Le"));
        _phase_function = scene->load_phase_function(desc->property_node_or_default("phasefunction"));
        LUISA_ASSERT(_sigma_a == nullptr || _sigma_a->is_constant(), "sigma_a must be constant");
        LUISA_ASSERT(_sigma_s == nullptr || _sigma_s->is_constant(), "sigma_s must be constant");
        LUISA_ASSERT(_Le == nullptr || _Le->is_constant(), "Le must be constant");
        LUISA_ASSERT(_phase_function != nullptr, "Phase function must be specified");
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::HomogeneousMedium)