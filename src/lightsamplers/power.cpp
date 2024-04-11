//
// Created by Leon Kang on 2024/4/9.
//

#include <util/sampling.h>
#include <base/light_sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace compute;

class PowerLightSampler final : public LightSampler {

private:
    float _environment_weight{.5f};

public:
    PowerLightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
        : LightSampler{scene, desc},
          _environment_weight{desc->property_float_or_default("environment_weight", .5f)} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto environment_weight() const noexcept { return _environment_weight; }
};

class PowerLightSamplerInstance final : public LightSampler::Instance {

private:
    luisa::unique_ptr<Shader1D<uint>> _compute_power;
    uint _light_handle_buffer_id{0u};
    uint _tag_lut_buffer_id{0u};
    Buffer<AliasEntry> _alias_table_buffer;
    Buffer<float> _pdf_buffer;
    float _env_prob{0.f};

public:
    PowerLightSamplerInstance(const PowerLightSampler *sampler, Pipeline &pipeline, CommandBuffer &command_buffer) noexcept
        : LightSampler::Instance{pipeline, sampler} {
        if (!pipeline.lights().empty()) {
            auto num_inst = static_cast<uint>(pipeline.geometry()->instances().size());
            auto num_light_inst = static_cast<uint>(pipeline.geometry()->light_instances().size());
            auto [light_handle_buffer_view, light_handle_buffer_id] = pipeline.bindless_arena_buffer<Light::Handle>(num_light_inst);
            _light_handle_buffer_id = light_handle_buffer_id;
            auto [tag_lut_buffer_view, tag_lut_buffer_id] = pipeline.bindless_arena_buffer<uint>(num_inst);
            _tag_lut_buffer_id = tag_lut_buffer_id;
            _alias_table_buffer = pipeline.device().create_buffer<AliasEntry>(num_light_inst);
            _pdf_buffer = pipeline.device().create_buffer<float>(num_light_inst);
            command_buffer << light_handle_buffer_view.copy_from(pipeline.geometry()->light_instances().data()) << commit();
            luisa::vector<uint> tag_lut(num_inst);
            for (auto i = 0u; i < num_light_inst; i++) {
                auto const &handle = pipeline.geometry()->light_instances()[i];
                tag_lut[handle.instance_id] = i;
            }
            command_buffer << tag_lut_buffer_view.copy_from(tag_lut.data()) << commit();
            _compute_power = luisa::make_unique<Shader1D<uint>>(pipeline.device().compile<1>([&](UInt num_light_inst) noexcept {
                set_block_size(256u);
                auto instance_id = def(0u);
                auto primitive_id = dispatch_id().x;
                $while(instance_id < num_light_inst) {
                    auto handle = light_handle_buffer_view->read(instance_id);
                    auto light_inst = pipeline.geometry()->instance(handle.instance_id);
                    $if(primitive_id >= light_inst.triangle_count()) {
                        instance_id += 1u;
                        primitive_id -= light_inst.triangle_count();
                        $continue;
                    } $else {
                        $break;
                    };
                };
                $while(instance_id < num_light_inst) {
                    auto handle = light_handle_buffer_view->read(instance_id);
                    auto light_inst = pipeline.geometry()->instance(handle.instance_id);
                    $if(primitive_id >= light_inst.triangle_count()) {
                        instance_id += 1u;
                        primitive_id -= light_inst.triangle_count();
                        $continue;
                    };
                    auto triangle = pipeline.geometry()->triangle(light_inst, primitive_id);
                    auto v_buffer = light_inst.vertex_buffer_id();
                    auto v0 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i0);
                    auto v1 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i1);
                    auto v2 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i2);
                    auto p0 = v0->position(), p1 = v1->position(), p2 = v2->position();
                    auto dp0 = p0 - p1, dp1 = p1 - p2;
                    auto object_to_world = pipeline.geometry()->instance_to_world(handle.instance_id);
                    auto m = make_float3x3(object_to_world);
                    auto c = cross(m * dp0, m * dp1);
                    auto surface_area = length(c) * .5f;
                    pipeline.lights().dispatch(light_inst.light_tag(), [&](auto light) noexcept {
                        auto power = surface_area * light->emission_power();
                        _pdf_buffer->atomic(instance_id).fetch_add(power);
                    });
                    primitive_id += dispatch_size().x;
                };
            }));
            luisa::vector<float> power_table(num_light_inst);
            command_buffer << (*_compute_power)(num_light_inst).dispatch(1024u)
                            << _pdf_buffer.copy_to(power_table.data())
                            << commit();
            auto [alias_table, pdf] = create_alias_table(power_table);
            command_buffer << _alias_table_buffer.copy_from(alias_table.data())
                            << _pdf_buffer.copy_from(pdf.data())
                            << commit();
        }
        if (pipeline.environment() != nullptr) {
            _env_prob = pipeline.lights().empty() ? 1.f : std::clamp(sampler->environment_weight(), 0.01f, 0.99f);
        }
    }

    void update(CommandBuffer &command_buffer) noexcept override {
        if (!pipeline().lights().empty()) {
            auto num_light_inst = static_cast<uint>(pipeline().geometry()->light_instances().size());

            luisa::vector<float> power_table(num_light_inst);
            command_buffer << (*_compute_power)(num_light_inst).dispatch(1024u)
                           << _pdf_buffer.copy_to(power_table.data())
                           << commit();
            command_buffer << synchronize();
            auto [alias_table, pdf] = create_alias_table(power_table);
            command_buffer << _alias_table_buffer.copy_from(alias_table.data())
                           << _pdf_buffer.copy_from(pdf.data())
                           << commit();
        }
    }

    [[nodiscard]] Float evaluate_selection(
        Expr<uint> tag, Expr<float3> p_from,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto prob = def(0.f);
        $if(tag == LightSampler::selection_environment) {
            prob = _env_prob;
        } $else {
            if (pipeline().lights().empty()) [[unlikely]] {// no lights
                LUISA_WARNING_WITH_LOCATION("No lights in scene.");
                prob = 0.f;
            } else {
                prob = (1.f - _env_prob) * _pdf_buffer->read(tag);
            }
        };
        return prob;
    }

    [[nodiscard]] Light::Evaluation evaluate_hit(
    const Interaction &it, Expr<float3> p_from,
    const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto eval = Light::Evaluation::zero(swl.dimension());
        if (pipeline().lights().empty()) [[unlikely]] {// no lights
            LUISA_WARNING_WITH_LOCATION("No lights in scene.");
            return eval;
        }
        pipeline().lights().dispatch(it.shape().light_tag(), [&](auto light) noexcept {
            auto closure = light->closure(swl, time);
            eval = closure->evaluate(it, p_from);
        });
        auto tag = pipeline().buffer<uint>(_tag_lut_buffer_id).read(it.instance_id());
        auto pdf_light = _pdf_buffer->read(tag);
        eval.pdf *= (1.f - _env_prob) * pdf_light;
        return eval;
    }

    [[nodiscard]] Light::Evaluation evaluate_miss(
    Expr<float3> wi, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        if (_env_prob == 0.f) [[unlikely]] {// no environment
            LUISA_WARNING_WITH_LOCATION("No environment in scene");
            return {.L = SampledSpectrum{swl.dimension()}, .pdf = 0.f};
        }
        auto eval = pipeline().environment()->evaluate(wi, swl, time);
        eval.pdf *= _env_prob;
        return eval;
    }

    [[nodiscard]] LightSampler::Selection select(
    const Interaction &it_from, Expr<float> u,
    const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        LUISA_ASSERT(pipeline().has_lighting(), "No lights in scene.");
        auto n = static_cast<float>(pipeline().geometry()->light_instances().size());
        if (_env_prob == 1.f) { return {.tag = LightSampler::selection_environment, .prob = 1.f}; }
        auto uu = (u - _env_prob) / (1.f - _env_prob);
        auto [tag, _] = sample_alias_table(_alias_table_buffer, static_cast<uint>(n), uu);
        auto prob = _pdf_buffer->read(tag);
        auto is_env = u < _env_prob;
        return {.tag = ite(is_env, LightSampler::selection_environment, tag),
                .prob = ite(is_env, _env_prob, (1.f - _env_prob) * prob)};
    }

    [[nodiscard]] LightSampler::Selection select(
    Expr<float> u,
    const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        LUISA_ASSERT(pipeline().has_lighting(), "No lights in scene.");
        auto n = static_cast<float>(pipeline().geometry()->light_instances().size());
        if (_env_prob == 1.f) { return {.tag = LightSampler::selection_environment, .prob = 1.f}; }
        auto uu = (u - _env_prob) / (1.f - _env_prob);
        auto [tag, _] = sample_alias_table(_alias_table_buffer, static_cast<uint>(n), uu);
        auto prob = _pdf_buffer->read(tag);
        auto is_env = u < _env_prob;
        return {.tag = ite(is_env, LightSampler::selection_environment, tag),
                .prob = ite(is_env, _env_prob, (1.f - _env_prob) * prob)};
    }

private:
    [[nodiscard]] auto _sample_area(Expr<float3> p_from,
                                    Expr<uint> tag,
                                    Expr<float2> u_in) const noexcept {
        auto handle = pipeline().buffer<Light::Handle>(_light_handle_buffer_id).read(tag);
        auto light_inst = pipeline().geometry()->instance(handle.instance_id);
        auto light_to_world = pipeline().geometry()->instance_to_world(handle.instance_id);
        auto alias_table_buffer_id = light_inst.alias_table_buffer_id();
        auto [triangle_id, ux] = sample_alias_table(
            pipeline().buffer<AliasEntry>(alias_table_buffer_id),
            light_inst.triangle_count(), u_in.x);
        auto triangle = pipeline().geometry()->triangle(light_inst, triangle_id);
        auto uvw = sample_uniform_triangle(make_float2(ux, u_in.y));
        auto attrib = pipeline().geometry()->shading_point(light_inst, triangle, uvw, light_to_world);
        return luisa::make_shared<Interaction>(std::move(light_inst), handle.instance_id,
                                               triangle_id, std::move(attrib),
                                               dot(attrib.g.n, p_from - attrib.g.p) < 0.f);
    }

    [[nodiscard]] Light::Sample _sample_light(const Interaction &it_from,
                                              Expr<uint> tag, Expr<float2> u,
                                              const SampledWavelengths &swl,
                                              Expr<float> time) const noexcept override {
        LUISA_ASSERT(!pipeline().lights().empty(), "No lights in the scene.");
        auto it = _sample_area(it_from.p(), tag, u);
        auto eval = Light::Evaluation::zero(swl.dimension());
        pipeline().lights().dispatch(it->shape().light_tag(), [&](auto light) noexcept {
            auto closure = light->closure(swl, time);
            eval = closure->evaluate(*it, it_from.p_shading());
        });
        return {.eval = std::move(eval), .p = it->p()};
    }

    [[nodiscard]] Environment::Sample _sample_environment(Expr<float2> u,
                                                          const SampledWavelengths &swl,
                                                          Expr<float> time) const noexcept override {
        LUISA_ASSERT(pipeline().environment() != nullptr, "No environment in the scene.");
        return pipeline().environment()->sample(swl, time, u);
    }
    //sample single light for L_emit.
    [[nodiscard]] LightSampler::Sample _sample_light_le(
        Expr<uint> tag, Expr<float2> u_light, Expr<float2> u_direction,
        const SampledWavelengths &swl,
        Expr<float> time) const noexcept override {
        LUISA_ASSERT(!pipeline().lights().empty(), "No lights in the scene.");
        auto handle = pipeline().buffer<Light::Handle>(_light_handle_buffer_id).read(tag);
        auto light_inst = pipeline().geometry()->instance(handle.instance_id);
        auto sp = Light::Sample::zero(swl.dimension());
        Var<Ray> shadow_ray{};
        pipeline().lights().dispatch(light_inst.light_tag(), [&](auto light) noexcept {
            auto closure = light->closure(swl, time);
            auto [sp_tp, ray_tp] = closure->sample_le(handle.instance_id, u_light, u_direction);
            sp = sp_tp;
            shadow_ray = ray_tp;
        });
        return {.eval = sp.eval, .shadow_ray = shadow_ray};
    }
};

unique_ptr<LightSampler::Instance> PowerLightSampler::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<PowerLightSamplerInstance>(
        this, pipeline, command_buffer);
}

} // namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::PowerLightSampler)