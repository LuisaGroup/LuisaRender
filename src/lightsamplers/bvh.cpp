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
    Buffer<uint> _bvh_lut_buffer;
    Buffer<uint4> _bvh_bounds_buffer;
    Buffer<uint4> _bvh_cone_buffer;
    Buffer<float4> _bvh_primitive_buffer;
    luisa::unique_ptr<Shader1D<uint>> _update_primitives;
    float3 _world_max;
    float3 _world_min;
    float _env_prob{0.f};

    struct alignas(16u) Bounds {
        float3 p_min{std::numeric_limits<float>::max()};
        float3 p_max{-std::numeric_limits<float>::max()};
        [[nodiscard]] auto centroid() const noexcept { return .5f * (p_min + p_max); }
        [[nodiscard]] auto extent() const noexcept { return p_max - p_min; }
        void combine(float3 const &p) noexcept {
            p_min = min(p_min, p);
            p_max = max(p_max, p);
        }
        void combine(Bounds const &b) noexcept {
            p_min = min(p_min, b.p_min);
            p_max = max(p_max, b.p_max);
        }
        [[nodiscard]] static auto combine(Bounds const &b1, Bounds const &b2) noexcept {
            return Bounds {
                .p_min = min(b1.p_min, b2.p_min),
                .p_max = max(b1.p_max, b2.p_max)
            };
        }
        [[nodiscard]] static auto combine(Bounds const &b, float3 const &p) noexcept {
            return Bounds {
                .p_min = min(b.p_min, p),
                .p_max = max(b.p_max, p)
            };
        }
    };
    static_assert(sizeof(Bounds) == 32u, "Invalid.");
    struct alignas(16u) BVHPrimitive {
        float x_min;
        float y_min;
        float z_min;
        uint tag;
        float x_max;
        float y_max;
        float z_max;
        float power;
        [[nodiscard]] auto bounds() const noexcept {
            return Bounds {
                .p_min = make_float3(x_min, y_min, z_min),
                .p_max = make_float3(x_max, y_max, z_max),
            };
        }
    };
    static auto constexpr QUANTIZE_EPS = 1e-6f;
    struct alignas(32u) QuantizedBVHNode {
        uint quantized_x;
        uint quantized_y;
        uint quantized_z;
        float power;
        uint quantized_axis;
        uint quantized_theta;
        uint two_sided;
        uint tag;
        [[nodiscard]] auto quantized_p_min() const noexcept {
            return make_uint3(quantized_x, quantized_y, quantized_z) & 0xffffu;
        }
        [[nodiscard]] auto quantized_p_max() const noexcept {
            return make_uint3(quantized_x, quantized_y, quantized_z) >> 16u;
        }
        [[nodiscard]] auto quantized_bounds() const noexcept {
            return make_uint4(quantized_x, quantized_y, quantized_z, bit_cast<uint>(power));
        }
        [[nodiscard]] auto quantized_cone() const noexcept {
            return make_uint4(quantized_axis, quantized_theta, two_sided, tag);
        }
    };
    static_assert(sizeof(QuantizedBVHNode) == 32u, "Invalid.");
    struct BVHNode {
        Float3 p_min;
        Float3 p_max;
        Float power;
        Float3 axis;
        Float cos_theta_o;
        Float cos_theta_e;
        UInt tag;
        [[nodiscard]] static auto decode(Expr<uint4> quantized_bounds, Expr<uint4> quantized_cone, Expr<float3> world_min, Expr<float3> world_max) noexcept {
            auto p_min = make_float3(quantized_bounds.xyz() & 0xffffu) / 65535.f * (world_max - world_min + QUANTIZE_EPS) + world_min;
            auto p_max = make_float3(quantized_bounds.xyz() >> 16u) / 65535.f * (world_max - world_min + QUANTIZE_EPS) + world_min;
            auto power = quantized_bounds.w.as<float>();
            // TODO: decode axis and theta_o, theta_e
            auto tag = quantized_cone.w;
            return BVHNode {
                .p_min = p_min,
                .p_max = p_max,
                .power = power,
                .tag = tag
            };
        }
        [[nodiscard]] auto compute_weight(Expr<float3> p_from) const noexcept {
            auto squared_distance = [](Expr<float3> p, Expr<float3> q) noexcept {
                return dot(p - q, p - q);
            };
            auto centroid = .5f * (p_min + p_max);
            return power / (squared_distance(p_from, centroid) + 1e-6f);
        }
    };
    class BVHBuilder {
    private:
        Bounds _centroid_bounds;
        template<typename T>
        [[nodiscard]] static auto _expand_bits(T x) noexcept {
            x = (x * 0x00010001u) & 0xff0000ffu;
            x = (x * 0x00000101u) & 0x0f00f00fu;
            x = (x * 0x00000011u) & 0xc30c30c3u;
            x = (x * 0x00000005u) & 0x49249249u;
            return x;
        }
        [[nodiscard]] auto _morton_3d(float3 p) const noexcept {
            p = (p - _centroid_bounds.p_min) / (_centroid_bounds.extent() + QUANTIZE_EPS);
            auto pp = make_uint3(clamp(p * 1024.0f, 0.0f, 1023.0f));
            pp = _expand_bits(pp);
            return pp.x << 2u | pp.y << 1u | pp.z;
        }
        [[nodiscard]] static auto _quantize_bounds(Bounds const &bounds, Bounds const &world_bounds) noexcept {
            auto quantized_p_min = make_uint3((bounds.p_min - world_bounds.p_min) / (world_bounds.extent() + 1e-6f) * 65535.0f);
            auto quantized_p_max = make_uint3((bounds.p_max - world_bounds.p_min) / (world_bounds.extent() + 1e-6f) * 65535.0f);
            return quantized_p_max << 16u | quantized_p_min;
        }
    public:
        [[nodiscard]] luisa::vector<QuantizedBVHNode> build(luisa::span<BVHPrimitive> primitives, Bounds const &world_bounds) noexcept {
            auto num_light_inst = static_cast<uint>(primitives.size());
            auto num_node = 2u * next_pow2(num_light_inst) - 1u;
            auto num_internal_node = next_pow2(num_light_inst) - 1u;
            auto nodes = luisa::vector<QuantizedBVHNode>(num_node);
            _centroid_bounds = std::accumulate(primitives.begin(), primitives.end(), Bounds{}, [](auto b, auto p) noexcept {
                return Bounds::combine(b, p.bounds().centroid());
            });
            luisa::vector<std::pair<uint, uint>> morton_codes(num_light_inst);
            for (auto i = 0u; i < num_light_inst; i++) {
                morton_codes[i] = std::make_pair(_morton_3d(primitives[i].bounds().centroid()), i);
            }
            if (morton_codes.size() < 1024u) {
                std::sort(morton_codes.begin(), morton_codes.end());
            } else {
                std::sort(std::execution::par, morton_codes.begin(), morton_codes.end());
            }
            auto current_node_index = num_node;
            while (current_node_index != 0u) {
                current_node_index--;
                if (current_node_index < num_internal_node) {
                    auto left_child_index = 2u * current_node_index + 1u;
                    auto right_child_index = 2u * current_node_index + 2u;
                    auto &left_child = nodes[left_child_index];
                    auto &right_child = nodes[right_child_index];
                    auto quantized_bounds = max(left_child.quantized_p_max(), right_child.quantized_p_max()) << 16u |
                                            min(left_child.quantized_p_min(), right_child.quantized_p_min());
                    nodes[current_node_index] = QuantizedBVHNode {
                        .quantized_x = quantized_bounds.x,
                        .quantized_y = quantized_bounds.y,
                        .quantized_z = quantized_bounds.z,
                        .power = left_child.power + right_child.power,
                        .tag = 0xffffffffu
                    };
                } else if (current_node_index < num_internal_node + primitives.size()) {
                    auto primitive_id = morton_codes[current_node_index - num_internal_node].second;
                    auto const &primitive = primitives[primitive_id];
                    auto quantized_bounds = _quantize_bounds(primitive.bounds(), world_bounds);
                    nodes[current_node_index] = QuantizedBVHNode {
                        .quantized_x = quantized_bounds.x,
                        .quantized_y = quantized_bounds.y,
                        .quantized_z = quantized_bounds.z,
                        .power = primitive.power,
                        .tag = primitive_id,
                    };
                } else {
                    nodes[current_node_index] = QuantizedBVHNode {
                        .quantized_x = 0xffffu,
                        .quantized_y = 0xffffu,
                        .quantized_z = 0xffffu,
                        .power = 0.f,
                        .tag = 0xffffffffu
                    };
                }
            }
            return nodes;
        }
    };

public:
    BVHLightSamplerInstance(const BVHLightSampler *sampler, Pipeline &pipeline, CommandBuffer &command_buffer)
        : LightSampler::Instance{pipeline, sampler} {
        if (!pipeline.lights().empty()) {
            auto num_inst = static_cast<uint>(pipeline.geometry()->instances().size());
            auto num_light_inst = static_cast<uint>(pipeline.geometry()->light_instances().size());
            auto num_node = 2u * next_pow2(num_light_inst) - 1u;
            auto num_internal_node = next_pow2(num_light_inst) - 1u;
            auto [light_handle_buffer_view, light_handle_buffer_id] = pipeline.bindless_arena_buffer<Light::Handle>(num_light_inst);
            _light_handle_buffer_id = light_handle_buffer_id;
            auto [tag_lut_buffer_view, tag_lut_buffer_id] = pipeline.bindless_arena_buffer<uint>(num_inst);
            _tag_lut_buffer_id = tag_lut_buffer_id;
            _bvh_lut_buffer = pipeline.device().create_buffer<uint>(num_light_inst);
            _bvh_bounds_buffer = pipeline.device().create_buffer<uint4>(num_node);
            _bvh_cone_buffer = pipeline.device().create_buffer<uint4>(num_node);
            _bvh_primitive_buffer = pipeline.device().create_buffer<float4>(2u * num_light_inst);
            luisa::vector<uint> tag_lut(num_inst);
            for (auto i = 0u; i < num_light_inst; i++) {
                auto const &handle = pipeline.geometry()->light_instances()[i];
                tag_lut[handle.instance_id] = i;
            }
            command_buffer << light_handle_buffer_view.copy_from(pipeline.geometry()->light_instances().data());
            command_buffer << tag_lut_buffer_view.copy_from(tag_lut.data());
            command_buffer << commit();
            _update_primitives = luisa::make_unique<Shader1D<uint>>(pipeline.device().compile<1>([&](UInt num_light_inst) noexcept {
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
                    };
                    auto object_to_world = pipeline.geometry()->instance_to_world(handle.instance_id);
                    auto m = make_float3x3(object_to_world);
                    auto t = make_float3(object_to_world[3]);
                    auto triangle = pipeline.geometry()->triangle(light_inst, primitive_id);
                    auto v_buffer = light_inst.vertex_buffer_id();
                    auto v0 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i0);
                    auto v1 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i1);
                    auto v2 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i2);
                    auto p0 = m * v0->position() + t, p1 = m * v1->position() + t, p2 = m * v2->position() + t;
                    auto p_min = min(min(p0, p1), p2), p_max = max(max(p0, p1), p2);
                    _bvh_primitive_buffer->atomic(instance_id << 1u).x.fetch_min(p_min.x);
                    _bvh_primitive_buffer->atomic(instance_id << 1u).y.fetch_min(p_min.y);
                    _bvh_primitive_buffer->atomic(instance_id << 1u).z.fetch_min(p_min.z);
                    _bvh_primitive_buffer->atomic(instance_id << 1u | 1u).x.fetch_max(p_max.x);
                    _bvh_primitive_buffer->atomic(instance_id << 1u | 1u).y.fetch_max(p_max.y);
                    _bvh_primitive_buffer->atomic(instance_id << 1u | 1u).z.fetch_max(p_max.z);
                    auto dp0 = p1 - p0, dp1 = p2 - p0;
                    auto c = cross(dp0, dp1);
                    auto surface_area = length(c) * .5f;
                    pipeline.lights().dispatch(light_inst.light_tag(), [&](auto light) noexcept {
                        _bvh_primitive_buffer->atomic(instance_id << 1u | 1u).w.fetch_add(light->emission_power() * surface_area);
                    });
                    primitive_id += dispatch_size().x;
                };
            }));
            luisa::vector<BVHPrimitive> primitives(num_light_inst);
            command_buffer << synchronize();
            Clock clock;
            command_buffer << (*_update_primitives)(num_light_inst).dispatch(1024u)
                            << _bvh_primitive_buffer.copy_to(primitives.data())
                            << commit()
                            << synchronize();
            auto world_bounds = std::accumulate(primitives.begin(), primitives.end(), Bounds{}, [](auto bounds, auto primitive) noexcept {
                return Bounds::combine(bounds, primitive.bounds());
            });
            _world_min = world_bounds.p_min;
            _world_max = world_bounds.p_max;
            BVHBuilder builder;
            auto nodes = builder.build(primitives, world_bounds);
            luisa::vector<uint> bvh_lut; bvh_lut.resize_uninitialized(num_light_inst);
            for (auto i = 0u; i < num_light_inst; i++) {
                bvh_lut[nodes[num_internal_node + i].tag] = num_internal_node + i;
            }
            command_buffer << _bvh_lut_buffer.copy_from(bvh_lut.data());
            luisa::vector<uint4> bvh_bounds(num_node);
            std::transform(nodes.begin(), nodes.end(), bvh_bounds.begin(), [](auto const &node) noexcept {
                return node.quantized_bounds();
            });
            command_buffer << _bvh_bounds_buffer.copy_from(bvh_bounds.data());
            luisa::vector<uint4> bvh_cone(num_node);
            std::transform(nodes.begin(), nodes.end(), bvh_cone.begin(), [](auto const &node) noexcept {
                return node.quantized_cone();
            });
            command_buffer << _bvh_cone_buffer.copy_from(bvh_cone.data())
                           << commit()
                           << synchronize();
            LUISA_INFO("BVH built in {} ms.", clock.toc());
        }
        if (pipeline.environment() != nullptr) {
            _env_prob = pipeline.lights().empty() ? 1.f : std::clamp(sampler->environment_weight(), 0.01f, 0.99f);
        }
    }

    void update(CommandBuffer &command_buffer) noexcept override {
        if (!pipeline().lights().empty()) {
            command_buffer << synchronize();
            Clock clock;
            auto num_light_inst = static_cast<uint>(pipeline().geometry()->light_instances().size());
            auto num_node = 2u * next_pow2(num_light_inst) - 1u;
            auto num_internal_node = next_pow2(num_light_inst) - 1u;
            luisa::vector<BVHPrimitive> primitives(num_light_inst);
            command_buffer << (*_update_primitives)(num_light_inst).dispatch(1024u)
                           << _bvh_primitive_buffer.copy_to(primitives.data())
                           << commit()
                           << synchronize();
            auto world_bounds = std::accumulate(primitives.begin(), primitives.end(), Bounds{}, [](auto bounds, auto primitive) noexcept {
                return Bounds::combine(bounds, primitive.bounds());
            });
            _world_min = world_bounds.p_min;
            _world_max = world_bounds.p_max;
            BVHBuilder builder;
            auto nodes = builder.build(primitives, world_bounds);
            luisa::vector<uint> bvh_lut; bvh_lut.resize_uninitialized(num_light_inst);
            for (auto i = 0u; i < num_light_inst; i++) {
                bvh_lut[nodes[num_internal_node + i].tag] = num_internal_node + i;
            }
            command_buffer << _bvh_lut_buffer.copy_from(bvh_lut.data());
            luisa::vector<uint4> bvh_bounds(num_node);
            std::transform(nodes.begin(), nodes.end(), bvh_bounds.begin(), [](auto const &node) noexcept {
                return node.quantized_bounds();
            });
            command_buffer << _bvh_bounds_buffer.copy_from(bvh_bounds.data());
            luisa::vector<uint4> bvh_cone(num_node);
            std::transform(nodes.begin(), nodes.end(), bvh_cone.begin(), [](auto const &node) noexcept {
                return node.quantized_cone();
            });
            command_buffer << _bvh_cone_buffer.copy_from(bvh_cone.data())
                           << commit()
                           << synchronize();
            LUISA_INFO("BVH built in {} ms.", clock.toc());
        }
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
        auto current_node_index = _bvh_lut_buffer->read(tag);
        auto current_node_pdf = def(1.f - _env_prob);
        $while(current_node_index != 0u) {
            auto current_node = BVHNode::decode(
                _bvh_bounds_buffer->read(current_node_index),
                _bvh_cone_buffer->read(current_node_index),
                _world_min, _world_max);
            auto sibling_node_index = ite((current_node_index % 2u) == 1u, current_node_index + 1u, current_node_index - 1u);
            auto sibling_node = BVHNode::decode(
                _bvh_bounds_buffer->read(sibling_node_index),
                _bvh_cone_buffer->read(sibling_node_index),
                _world_min, _world_max);
            auto w1 = current_node.compute_weight(p_from), w2 = sibling_node.compute_weight(p_from);
            current_node_pdf *= w1 / (w1 + w2 + 1e-6f);
            current_node_index = (current_node_index - 1u) >> 1u;
        };
        eval.pdf *= current_node_pdf;
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
        if (_env_prob == 1.f) { return {.tag = LightSampler::selection_environment, .prob = 1.f}; }
        auto uu = (u - _env_prob) / (1.f - _env_prob);
        auto current_node_index = def(0u);
        auto current_node_pdf = def(1.f - _env_prob);
        auto num_light_inst = cast<uint>(pipeline().geometry()->light_instances().size());
        auto num_internal_nodes = next_pow2(num_light_inst) - 1u;
        $while(current_node_index < num_internal_nodes) {
            auto left_child_index = 2u * current_node_index + 1u;
            auto right_child_index = 2u * current_node_index + 2u;
            auto left_child_node = BVHNode::decode(
                _bvh_bounds_buffer->read(left_child_index),
                _bvh_cone_buffer->read(left_child_index),
                _world_min, _world_max);
            auto right_child_node = BVHNode::decode(
                _bvh_bounds_buffer->read(right_child_index),
                _bvh_cone_buffer->read(right_child_index),
                _world_min, _world_max);
            $if(right_child_node.power == 0.f) {
                current_node_index = left_child_index;
                $continue;
            } $elif(left_child_node.power == 0.f) {
                current_node_index = right_child_index;
                $continue;
            };
            auto w1 = left_child_node.compute_weight(it_from.p()), w2 = right_child_node.compute_weight(it_from.p());
            auto left_prob = w1 / (w1 + w2 + 1e-6f);
            $if(uu < left_prob) {
                uu /= left_prob;
                current_node_pdf *= left_prob;
                current_node_index = left_child_index;
            } $else {
                uu = (uu - left_prob) / (1.f - left_prob);
                current_node_pdf *= 1.f - left_prob;
                current_node_index = right_child_index;
            };
        };
        auto current_node = BVHNode::decode(
            _bvh_bounds_buffer->read(current_node_index),
            _bvh_cone_buffer->read(current_node_index),
            _world_min, _world_max);
        auto is_env = u < _env_prob;
        return {.tag = ite(is_env, LightSampler::selection_environment, current_node.tag),
                .prob = ite(is_env, _env_prob, current_node_pdf)};
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
