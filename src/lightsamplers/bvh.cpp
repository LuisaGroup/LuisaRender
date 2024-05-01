//
// Created by Leon Kang on 2024/4/9.
//

#include <algorithm>
#include <numeric>
#include <execution>

#include <util/sampling.h>
#include <base/light_sampler.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace compute;

class BVHLightSampler final : public LightSampler {

private:
    float _environment_weight{.5f};

public:
    BVHLightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept
        : LightSampler{scene, desc},
          _environment_weight{desc->property_float_or_default("environment_weight", .5f)} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] auto environment_weight() const noexcept { return _environment_weight; }
};

class BVHLightSamplerInstance final : public LightSampler::Instance {

private:
    uint _light_handle_buffer_id{0u};
    uint _tag_lut_buffer_id{0u};
    uint _bvh_node_buffer_id{0u};
    uint _bvh_lut_buffer_id{0u};
    float _env_prob;
    luisa::unique_ptr<Shader1D<uint>> _clear_vpl_list;
    luisa::unique_ptr<Shader1D<uint, float>> _update_vpl_list;
    luisa::unique_ptr<Shader1D<uint>> _compute_morton_code;
    luisa::unique_ptr<Shader1D<uint>> _update_bvh_lut;
    luisa::unique_ptr<Shader1D<uint>> _clear_atomic_counter;
    luisa::unique_ptr<Shader1D<uint, uint>> _update_bvh_node;
    BufferView<float4> _bvh_node_buffer_view;
    BufferView<uint> _bvh_lut_buffer_view;
    BufferView<float3> _object_space_aabb_buffer_view;
    BufferView<uint2> _morton_code_buffer_view;
    BufferView<float3> _world_bounds_buffer_view;
    BufferView<float4> _vpl_list_buffer_view;
    BufferView<uint> _atomic_counter_buffer_view;

public:
    BVHLightSamplerInstance(const BVHLightSampler *sampler, Pipeline &pipeline, CommandBuffer &command_buffer)
        : LightSampler::Instance{pipeline, sampler} {
        if (!pipeline.lights().empty()) {
            auto num_inst = static_cast<uint>(pipeline.geometry()->instances().size());
            auto num_light_inst = static_cast<uint>(pipeline.geometry()->light_instances().size());
            auto [light_handle_buffer_view, light_handle_buffer_id] = pipeline.bindless_arena_buffer<Light::Handle>(num_light_inst);
            _light_handle_buffer_id = light_handle_buffer_id;
            auto [tag_lut_buffer_view, tag_lut_buffer_id] = pipeline.bindless_arena_buffer<uint>(num_inst);
            _tag_lut_buffer_id = tag_lut_buffer_id;
            auto num_leaf_node = next_pow2(num_light_inst);
            auto num_internal_node = num_leaf_node - 1;
            auto num_node = num_leaf_node + num_internal_node;
            auto [bvh_node_buffer_view, bvh_node_buffer_id] = pipeline.bindless_arena_buffer<float4>(num_node);
            _bvh_node_buffer_id = bvh_node_buffer_id;
            _bvh_node_buffer_view = bvh_node_buffer_view;
            auto [bvh_lut_buffer_view, bvh_lut_buffer_id] = pipeline.bindless_arena_buffer<uint>(2u * num_light_inst);
            _bvh_lut_buffer_id = bvh_lut_buffer_id;
            _bvh_lut_buffer_view = bvh_lut_buffer_view;
            auto [object_space_aabb_buffer_view, object_space_aabb_buffer_id] = pipeline.bindless_arena_buffer<float3>(2u * num_light_inst);
            _object_space_aabb_buffer_view = object_space_aabb_buffer_view;
            auto [world_bounds_buffer_view, world_bounds_buffer_id] = pipeline.bindless_arena_buffer<float3>(2u);
            _world_bounds_buffer_view = world_bounds_buffer_view;
            auto [vpl_list_buffer_view, vpl_list_buffer_id] = pipeline.bindless_arena_buffer<float4>(num_light_inst);
            _vpl_list_buffer_view = vpl_list_buffer_view;
            auto [morton_code_buffer_view, morton_code_buffer_id] = pipeline.bindless_arena_buffer<uint2>(num_light_inst);
            _morton_code_buffer_view = morton_code_buffer_view;
            auto [atomic_counter_buffer_view, atomic_counter_buffer_id] = pipeline.bindless_arena_buffer<uint>(num_internal_node);
            _atomic_counter_buffer_view = atomic_counter_buffer_view;
            command_buffer << light_handle_buffer_view.copy_from(pipeline.geometry()->light_instances().data()) << commit();
            luisa::vector<uint> tag_lut(num_inst);
            for (auto i = 0u; i < num_light_inst; i++) {
                auto const &handle = pipeline.geometry()->light_instances()[i];
                tag_lut[handle.instance_id] = i;
            }
            command_buffer << tag_lut_buffer_view.copy_from(tag_lut.data()) << commit();
            auto clear_object_space_aabb_buffer = pipeline.device().compile<1>([&](UInt n) noexcept {
                set_block_size(256u);
                auto i = dispatch_id().x;
                $while(i < n) {
                    _object_space_aabb_buffer_view->write(i, make_float3(std::numeric_limits<float>::max()));
                    _object_space_aabb_buffer_view->write(i + n, make_float3(-std::numeric_limits<float>::max()));
                    i += dispatch_size().x;
                };
            });
            auto compute_object_space_aabb = pipeline.device().compile<1>([&](UInt n) noexcept {
                set_block_size(256u);
                auto tag = def(0u);
                $while(tag < n) {
                    auto p_min = def<float3>(make_float3(std::numeric_limits<float>::max())), p_max = -p_min;
                    auto handle = light_handle_buffer_view->read(tag);
                    auto light_inst = pipeline.geometry()->instance(handle.instance_id);
                    auto primitive_id = dispatch_id().x;
                    $while(primitive_id < light_inst.triangle_count()) {
                        auto triangle = pipeline.geometry()->triangle(light_inst, primitive_id);
                        auto v_buffer = light_inst.vertex_buffer_id();
                        auto v0 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i0);
                        auto v1 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i1);
                        auto v2 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i2);
                        auto p0 = v0->position();
                        auto p1 = v1->position();
                        auto p2 = v2->position();
                        p_min = min(p_min, min(p0, min(p1, p2)));
                        p_max = max(p_max, max(p0, max(p1, p2)));
                        primitive_id += dispatch_size().x;
                    };
                    _object_space_aabb_buffer_view->atomic(tag).x.fetch_min(p_min.x);
                    _object_space_aabb_buffer_view->atomic(tag).y.fetch_min(p_min.y);
                    _object_space_aabb_buffer_view->atomic(tag).z.fetch_min(p_min.z);
                    _object_space_aabb_buffer_view->atomic(tag + n).x.fetch_max(p_max.x);
                    _object_space_aabb_buffer_view->atomic(tag + n).y.fetch_max(p_max.y);
                    _object_space_aabb_buffer_view->atomic(tag + n).z.fetch_max(p_max.z);
                    tag += 1u;
                };
            });
            _clear_vpl_list = luisa::make_unique<Shader1D<uint>>(pipeline.device().compile<1>([&](UInt n) noexcept {
                set_block_size(256u);
                auto i = dispatch_id().x;
                $if(i == 0u) {
                    _world_bounds_buffer_view->write(0u, make_float3(std::numeric_limits<float>::max()));
                    _world_bounds_buffer_view->write(1u, -make_float3(std::numeric_limits<float>::max()));
                };
                $while(i < n) {
                    _vpl_list_buffer_view->write(i, make_float4(0.f));
                    i += dispatch_size().x;
                };
            }));
            _update_vpl_list = luisa::make_unique<Shader1D<uint, float>>(pipeline.device().compile<1>([&](UInt n, Float time) noexcept {
                set_block_size(256u);
                auto instance_id = def(0u);
                $while(instance_id < n) {
                    auto power = def(0.f);
                    auto handle = light_handle_buffer_view->read(instance_id);
                    auto light_inst = pipeline.geometry()->instance(handle.instance_id);
                    auto object_to_world = pipeline.geometry()->instance_to_world(handle.instance_id);
                    auto m = make_float3x3(object_to_world);
                    auto t = make_float3(object_to_world[3]);
                    auto primitive_id = dispatch_id().x;
                    $if(primitive_id == 0u) {
                        auto centroid = (_object_space_aabb_buffer_view->read(instance_id) + _object_space_aabb_buffer_view->read(instance_id + n)) * .5f;
                        centroid = m * centroid + t;
                        _vpl_list_buffer_view->atomic(instance_id).x.exchange(centroid.x);
                        _vpl_list_buffer_view->atomic(instance_id).y.exchange(centroid.y);
                        _vpl_list_buffer_view->atomic(instance_id).z.exchange(centroid.z);
                        _world_bounds_buffer_view->atomic(0u).x.fetch_min(centroid.x);
                        _world_bounds_buffer_view->atomic(0u).y.fetch_min(centroid.y);
                        _world_bounds_buffer_view->atomic(0u).z.fetch_min(centroid.z);
                        _world_bounds_buffer_view->atomic(1u).x.fetch_max(centroid.x);
                        _world_bounds_buffer_view->atomic(1u).y.fetch_max(centroid.y);
                        _world_bounds_buffer_view->atomic(1u).z.fetch_max(centroid.z);
                    };
                    $while(primitive_id < light_inst.triangle_count()) {
                        auto triangle = pipeline.geometry()->triangle(light_inst, primitive_id);
                        auto v_buffer = light_inst.vertex_buffer_id();
                        auto v0 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i0);
                        auto v1 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i1);
                        auto v2 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i2);
                        auto dp0 = m * (v1->position() - v0->position()), dp1 = m * (v2->position() - v0->position());
                        auto surface_area = length(cross(dp0, dp1)) * .5f;
                        auto emission_luminance = def(0.f);
                        pipeline.lights().dispatch(light_inst.light_tag(), [&](auto light) noexcept {
                            auto closure = light->closure(SampledWavelengths{pipeline.spectrum()->sample(0.f)}, time);
                            emission_luminance += closure->evaluate_luminance(Interaction(v0->uv()));
                            emission_luminance += closure->evaluate_luminance(Interaction(v1->uv()));
                            emission_luminance += closure->evaluate_luminance(Interaction(v2->uv()));
                            emission_luminance *= 2.f / 3.f;
                            auto gravity_center_uv = (v0->uv() + v1->uv() + v2->uv()) / 3.f;
                            emission_luminance += closure->evaluate_luminance(Interaction(gravity_center_uv)) / 3.f;
                        });
                        power += emission_luminance * surface_area;
                        primitive_id += dispatch_size().x;
                    };
                    _vpl_list_buffer_view->atomic(instance_id).w.fetch_add(power);
                    instance_id += 1u;
                };
            }));
            _compute_morton_code = luisa::make_unique<Shader1D<uint>>(pipeline.device().compile<1>([&](UInt n) noexcept {
                set_block_size(256u);
                auto i = dispatch_id().x;
                auto world_min = _world_bounds_buffer_view->read(0u);
                auto world_max = _world_bounds_buffer_view->read(1u);
                $while(i < n) {
                    auto p = _vpl_list_buffer_view->read(i).xyz();
                    auto x = make_uint3(clamp((p - world_min) / (world_max - world_min + 1e-6f) * 1024.f, 0.f, 1023.f));
                    x = (x * 0x00010001u) & 0xFF0000FFu;
                    x = (x * 0x00000101u) & 0x0F00F00Fu;
                    x = (x * 0x00000011u) & 0xC30C30C3u;
                    x = (x * 0x00000005u) & 0x49249249u;
                    auto morton_code = 4u * x.z + 2u * x.y + x.x;
                    _morton_code_buffer_view->write(i, make_uint2(morton_code, i));
                    i += dispatch_size().x;
                };
            }));
            _update_bvh_lut = luisa::make_unique<Shader1D<uint>>(pipeline.device().compile<1>([&](UInt n) noexcept {
                set_block_size(256u);
                auto i = dispatch_id().x;
                $while(i < n) {
                    auto instance_id = _morton_code_buffer_view->read(i).y;
                    _bvh_lut_buffer_view->write(i, instance_id);
                    _bvh_lut_buffer_view->write(instance_id + n, i);
                    i += dispatch_size().x;
                };
            }));
            _clear_atomic_counter = luisa::make_unique<Shader1D<uint>>(pipeline.device().compile<1>([&](UInt n) noexcept {
                set_block_size(256u);
                auto i = dispatch_id().x;
                $while(i < n) {
                    _atomic_counter_buffer_view->write(i, 0u);
                    i += dispatch_size().x;
                };
            }));
            _update_bvh_node = luisa::make_unique<Shader1D<uint, uint>>(pipeline.device().compile<1>([&](UInt num_internal_node, UInt num_light_inst) noexcept {
                set_block_size(256u);
                auto leaf_id = dispatch_id().x;
                auto current = num_internal_node + leaf_id;
                auto leaf_node = def<float4>(make_float4(0.f));
                $if(leaf_id < num_light_inst) {
                    auto vpl_id = _bvh_lut_buffer_view->read(leaf_id);
                    leaf_node = _vpl_list_buffer_view->read(vpl_id);
                };
                _bvh_node_buffer_view->write(current, leaf_node);
                $loop {
                    $if(current == 0u) {
                        $break;
                    };
                    current = (current - 1u) / 2u;
                    auto atomic_flag = _atomic_counter_buffer_view->atomic(current).fetch_add(1u);
                    $if(atomic_flag == 0u) { $break; };
                    auto left_node = _bvh_node_buffer_view->read(2u * current + 1u);
                    auto right_node = _bvh_node_buffer_view->read(2u * current + 2u);
                    auto power = left_node.w + right_node.w;
                    auto centroid = def<float3>(make_float3(0.f));
                    $if(power > 0.f) {
                        centroid = (left_node.xyz() * left_node.w + right_node.xyz() * right_node.w) / power;
                    };
                    _bvh_node_buffer_view->write(current, make_float4(centroid, power));
                };
            }));
            command_buffer << synchronize();
            command_buffer << clear_object_space_aabb_buffer(num_light_inst).dispatch(1024u)
                           << compute_object_space_aabb(num_light_inst).dispatch(1024u)
                           << commit();
        }
        if (pipeline.environment() != nullptr) {
            _env_prob = pipeline.lights().empty() ? 1.f : std::clamp(sampler->environment_weight(), 0.01f, 0.99f);
        }
    }

    void update(CommandBuffer &command_buffer, float time) noexcept override {
        if (!pipeline().lights().empty()) {
            command_buffer << synchronize();
            Clock clk;
            auto n = static_cast<uint>(pipeline().geometry()->light_instances().size());
            auto num_leaf_node = next_pow2(n);
            auto num_internal_node = num_leaf_node - 1;
            command_buffer << (*_clear_vpl_list)(n).dispatch(1024u)
                           << (*_update_vpl_list)(n, time).dispatch(1024u)
                           << (*_compute_morton_code)(n).dispatch(1024u)
                           << commit();
            luisa::vector<uint2> morton_code_list(n);
            command_buffer << _morton_code_buffer_view.copy_to(morton_code_list.data()) << commit();
            command_buffer << synchronize();
            std::sort(morton_code_list.begin(), morton_code_list.end(), [](auto a, auto b) noexcept {
                return a.x < b.x;
            });
            command_buffer << _morton_code_buffer_view.copy_from(morton_code_list.data())
                           << (*_update_bvh_lut)(n).dispatch(1024u)
                           << (*_clear_atomic_counter)(n).dispatch(1024u)
                           << (*_update_bvh_node)(num_internal_node, n).dispatch(num_leaf_node)
                           << commit();
            command_buffer << synchronize();
            LUISA_INFO("BVH updated in {} ms.", clk.toc());
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
                auto n = static_cast<uint>(pipeline().geometry()->light_instances().size());
                auto num_leaf_node = next_pow2(n);
                auto num_internal_node = num_leaf_node - 1u;
                prob = (1.f - _env_prob);
                auto current = num_internal_node + pipeline().buffer<uint>(_bvh_lut_buffer_id).read(tag + n);
                $while(current != 0u) {
                    auto sibling = ite((current & 1u) == 1u, current + 1u, current - 1u);
                    auto current_node = pipeline().buffer<float4>(_bvh_node_buffer_id).read(current);
                    auto sibling_node = pipeline().buffer<float4>(_bvh_node_buffer_id).read(sibling);
                    auto w1 = current_node.w / (distance_squared(current_node.xyz(), p_from) + 1e-6f),
                         w2 = sibling_node.w / (distance_squared(sibling_node.xyz(), p_from) + 1e-6f);
                    prob *= w1 / (w1 + w2);
                    current = (current - 1u) / 2u;
                };
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
        eval.pdf *= evaluate_selection(tag, p_from, swl, time);
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
        auto n = static_cast<uint>(pipeline().geometry()->light_instances().size());
        auto num_leaf_node = next_pow2(n);
        auto num_internal_node = num_leaf_node - 1u;
        if (_env_prob == 1.f) { return {.tag = LightSampler::selection_environment, .prob = 1.f}; }
        auto uu = (u - _env_prob) / (1.f - _env_prob);
        auto current = def(0u);
        auto prob = def<float>(1.f - _env_prob);
        $while(current < num_internal_node) {
            auto left_node = pipeline().buffer<float4>(_bvh_node_buffer_id).read(2u * current + 1u);
            auto right_node = pipeline().buffer<float4>(_bvh_node_buffer_id).read(2u * current + 2u);
            $if(right_node.w == 0.f) {
                current = 2u * current + 1u;
                $continue;
            };
            auto w1 = left_node.w / (distance_squared(left_node.xyz(), it_from.p()) + 1e-6f),
                 w2 = right_node.w / (distance_squared(right_node.xyz(), it_from.p()) + 1e-6f);
            auto left_prob = w1 / (w1 + w2);
            $if(uu < left_prob) {
                current = 2u * current + 1u;
                prob *= left_prob;
                uu /= left_prob;
            } $else {
                current = 2u * current + 2u;
                prob *= 1.f - left_prob;
                uu = (uu - left_prob) / (1.f - left_prob);
            };
        };
        auto is_env = u < _env_prob | current >= num_internal_node + n;
        return {.tag = ite(is_env,
                           LightSampler::selection_environment,
                           pipeline().buffer<uint>(_bvh_lut_buffer_id).read(current - num_internal_node)),
                .prob = ite(is_env, _env_prob, prob)};
    }

    [[nodiscard]] LightSampler::Selection select(
        Expr<float> u, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        LUISA_WARNING_WITH_LOCATION("BVHLightSampler does not support light selection independent of shading point.");
        return {.tag = LightSampler::selection_environment, .prob = 1.f};
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

unique_ptr<LightSampler::Instance> BVHLightSampler::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<BVHLightSamplerInstance>(
        this, pipeline, command_buffer);
}

} // namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::BVHLightSampler)
