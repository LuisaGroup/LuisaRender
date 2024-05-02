//
// Created by Leon Kang on 2024/4/9.
//

#include <util/sampling.h>
#include <base/light_sampler.h>
#include <base/pipeline.h>

namespace luisa::render {
    struct BVHPrimitive {
        float x_min;
        float y_min;
        float z_min;
        float x_max;
        float y_max;
        float z_max;
        float power{0.f};
        uint tag{0xffffffff};
        [[nodiscard]] float3 p_min() const noexcept { return make_float3(x_min, y_min, z_min); }
        [[nodiscard]] float3 p_max() const noexcept { return make_float3(x_max, y_max, z_max); }
    };
    class BVHNode {
    public:
        float3 p_min;
        float3 p_max;
        float power;
        uint first_primitive_offset;
        uint primitive_count;
    private:
        luisa::unique_ptr<BVHNode> _left_child;
        luisa::unique_ptr<BVHNode> _right_child;
    public:
        BVHNode(float3 p_min, float3 p_max, float power, uint first_primitive_offset, uint primitive_count) noexcept
            : p_min(p_min), p_max(p_max), power(power), first_primitive_offset(first_primitive_offset), primitive_count(primitive_count) {}
        BVHNode(luisa::unique_ptr<BVHNode> left_child, luisa::unique_ptr<BVHNode> right_child) noexcept {
            if (!left_child || !right_child) {LUISA_ERROR_WITH_LOCATION("Invalid BVH child node.");}
            _left_child = std::move(left_child);
            _right_child = std::move(right_child);
            p_min = min(_left_child->p_min, _right_child->p_min);
            p_max = max(_left_child->p_max, _right_child->p_max);
            power = _left_child->power + _right_child->power;
            first_primitive_offset = min(_left_child->first_primitive_offset, _right_child->first_primitive_offset);
            primitive_count = _left_child->primitive_count + _right_child->primitive_count;
        }
        [[nodiscard]] uint size() const noexcept {
            return _left_child ? (_left_child->size() + _right_child->size() + 1u) : 1u;
        }
        [[nodiscard]] bool is_leaf() const noexcept { return !_left_child; }
        [[nodiscard]] BVHNode *left_child() const noexcept { return _left_child.get(); }
        [[nodiscard]] BVHNode *right_child() const noexcept { return _right_child.get(); }
    };
    enum BVHSplitAxis {
        X = 0u,
        Y = 1u,
        Z = 2u
    };
    class BVHSplitBin {
    private:
        float3 _p_min;
        float3 _p_max;
        float _power;
        luisa::list<BVHPrimitive> _primitives;
    public:
        BVHSplitBin() noexcept {
            _p_min = make_float3(std::numeric_limits<float>::max());
            _p_max = make_float3(std::numeric_limits<float>::lowest());
            _power = 0.f;
        }
        [[nodiscard]] auto p_min() const noexcept { return _p_min; }
        [[nodiscard]] auto p_max() const noexcept { return _p_max; }
        [[nodiscard]] auto power() const noexcept { return _power; }
        [[nodiscard]] auto &primitives() noexcept { return _primitives; }
        void update() noexcept {
            _p_min = make_float3(std::numeric_limits<float>::max());
            _p_max = make_float3(std::numeric_limits<float>::lowest());
            _power = 0.f;
            for (auto const &primitive : _primitives) {
                _p_min = min(_p_min, primitive.p_min());
                _p_max = max(_p_max, primitive.p_max());
                _power += primitive.power;
            }
        }
    };
    [[nodiscard]] luisa::unique_ptr<BVHNode> create_bvh(luisa::list<BVHPrimitive> &primitives, uint first_primitive_offset = 0u) noexcept {
        auto constexpr MAX_LEAF_CAPACITY = 4u;
        auto create_leaf_node = [&]() noexcept {
            auto p_min = make_float3(std::numeric_limits<float>::max()), p_max = make_float3(std::numeric_limits<float>::lowest());
            auto power = 0.f;
            for (auto const &primitive : primitives) {
                p_min = min(p_min, primitive.p_min());
                p_max = max(p_max, primitive.p_max());
                power += primitive.power;
            }
            return luisa::make_unique<BVHNode>(p_min, p_max, power, first_primitive_offset, primitives.size());
        };
        if (primitives.size() <= MAX_LEAF_CAPACITY) {
            return create_leaf_node();
        }
        auto constexpr NUM_BINS = 12u;
        auto constexpr NUM_SPLITS = NUM_BINS - 1u;
        std::array<BVHSplitBin, NUM_BINS> bins;
        auto constexpr compute_split_axis = [](auto extent) noexcept {
            if (extent.x > extent.y) {
                return extent.x > extent.z ? BVHSplitAxis::X : BVHSplitAxis::Z;
            } else {
                return extent.y > extent.z ? BVHSplitAxis::Y : BVHSplitAxis::Z;
            }
        };
        auto range_min = make_float3(std::numeric_limits<float>::max());
        auto range_max = make_float3(std::numeric_limits<float>::lowest());
        for (auto const &primitive : primitives) {
            range_min = min(range_min, primitive.p_min());
            range_max = max(range_max, primitive.p_max());
        }
        auto range_extent = range_max - range_min;
        auto split_axis = compute_split_axis(range_extent);
        while (!primitives.empty()) {
            auto const &primitive = primitives.front();
            auto centroid = (primitive.p_min() + primitive.p_max()) * .5f;
            auto bin_index = static_cast<uint>(clamp((centroid[split_axis] - range_min[split_axis]) / range_extent[split_axis] * NUM_BINS, 0.f, NUM_BINS - 1.f));
            auto &bin = bins[bin_index];
            bin.primitives().splice(bin.primitives().end(), primitives, primitives.begin());
        }
        for (auto &bin : bins) {bin.update();}
        float split_costs[NUM_SPLITS];
        std::fill(split_costs, split_costs + NUM_SPLITS, 0.f);
        auto split_min = make_float3(std::numeric_limits<float>::max());
        auto split_max = make_float3(std::numeric_limits<float>::lowest());
        auto split_power = 0.f;
        for (auto i = 0u; i < NUM_SPLITS; i++) {
            split_min = min(split_min, bins[i].p_min());
            split_max = max(split_max, bins[i].p_max());
            split_power += bins[i].power();
            auto split_extent = split_max - split_min;
            auto split_area = 2.f * (split_extent.x * split_extent.y + split_extent.y * split_extent.z + split_extent.z * split_extent.x);
            split_costs[i] += split_power * split_area;
        }
        split_min = make_float3(std::numeric_limits<float>::max());
        split_max = make_float3(std::numeric_limits<float>::lowest());
        split_power = 0.f;
        for (auto i = NUM_SPLITS; i > 0u; i--) {
            split_min = min(split_min, bins[i].p_min());
            split_max = max(split_max, bins[i].p_max());
            split_power += bins[i].power();
            auto split_extent = split_max - split_min;
            auto split_area = 2.f * (split_extent.x * split_extent.y + split_extent.y * split_extent.z + split_extent.z * split_extent.x);
            split_costs[i - 1u] += split_power * split_area;
        }
        auto min_split_cost = std::numeric_limits<float>::max();
        auto best_split_index = 0u;
        for (auto i = 0u; i < NUM_SPLITS; i++) {
            if (split_costs[i] < min_split_cost) {
                min_split_cost = split_costs[i];
                best_split_index = i;
            }
        }
        luisa::list<BVHPrimitive> left_child_primitives, right_child_primitives;
        for (auto i = 0u; i < NUM_BINS; i++) {
            if (i <= best_split_index) {
                left_child_primitives.splice(left_child_primitives.end(), bins[i].primitives());
            } else {
                right_child_primitives.splice(right_child_primitives.end(), bins[i].primitives());
            }
        }
        if (left_child_primitives.empty() | right_child_primitives.empty()) {
            primitives.splice(primitives.end(), left_child_primitives);
            primitives.splice(primitives.end(), right_child_primitives);
            return create_leaf_node();
        }
        auto left_child = create_bvh(left_child_primitives, first_primitive_offset);
        auto right_child = create_bvh(right_child_primitives, first_primitive_offset + left_child_primitives.size());
        primitives.splice(primitives.end(), left_child_primitives);
        primitives.splice(primitives.end(), right_child_primitives);
        return luisa::make_unique<BVHNode>(std::move(left_child), std::move(right_child));
    }
    struct alignas(16u) QBVHNode {
        uint quantized_p_min;
        uint quantized_p_max;
        float power;
        uint parent;
        uint right_child;
        uint first_primitive_offset;
        uint primitive_count;
        [[nodiscard]] static auto encode(const BVHNode &node, float3 world_min, float3 world_max) noexcept {
            auto constexpr QUANTIZED_COORDINATE_BITS = 10u;
            auto constexpr MAX_COORDINATE_VALUE = static_cast<float>((1u << QUANTIZED_COORDINATE_BITS) - 1u);
            auto encode_coordinates = [&](auto p) noexcept {
                auto x = make_uint3(clamp((p - world_min) / (world_max - world_min + std::numeric_limits<float>::epsilon()) * MAX_COORDINATE_VALUE, 0.f, MAX_COORDINATE_VALUE));
                return x.x | (x.y << QUANTIZED_COORDINATE_BITS) | (x.z << (2u * QUANTIZED_COORDINATE_BITS));
            };
            return QBVHNode {
                .quantized_p_min = encode_coordinates(node.p_min),
                .quantized_p_max = encode_coordinates(node.p_max),
                .power = node.power,
                .parent = 0xffffffffu,
                .right_child = 0xffffffffu,
                .first_primitive_offset = node.first_primitive_offset,
                .primitive_count = node.primitive_count
            };
        }
    };
    static_assert(sizeof(BVHPrimitive) == 32u, "Invalid.");
    static_assert(sizeof(QBVHNode) == 32u, "Invalid.");
    [[nodiscard]] luisa::vector<QBVHNode> encode_quantized_bvh(BVHNode *root, float3 world_min, float3 world_max) noexcept {
        luisa::vector<QBVHNode> quantized_nodes;
        quantized_nodes.reserve(root->size());
        luisa::stack<std::pair<BVHNode *, uint>> stack; /* child node, parent node index */
        stack.push({root, 0xffffffffu});
        while (!stack.empty()) {
            auto [node, parent_index] = stack.top();
            stack.pop();
            auto current_index = static_cast<uint>(quantized_nodes.size());
            if (parent_index != 0xffffffffu) {
                quantized_nodes[parent_index].parent = current_index;
                if (current_index != parent_index + 1u) {
                    quantized_nodes[parent_index].right_child = current_index;
                }
            }
            quantized_nodes.push_back(QBVHNode::encode(*node, world_min, world_max));
            if (!node->is_leaf()) {
                stack.push({node->right_child(), current_index});
                stack.push({node->left_child(), current_index}); /* left child is traversed right after the current node, hence no need to store left child's index */
            }
        }
        return quantized_nodes;
    }
} // namespace luisa::render

LUISA_STRUCT(luisa::render::BVHPrimitive, x_min, y_min, z_min, x_max, y_max, z_max, power, tag){};
LUISA_STRUCT(luisa::render::QBVHNode, quantized_p_min, quantized_p_max, power, parent, right_child, first_primitive_offset, primitive_count){};

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
    luisa::shared_ptr<Shader1D<>> _clear_primitive_data;
    luisa::shared_ptr<Shader1D<float>> _update_primitive_data;
    luisa::shared_ptr<Buffer<float3>> _world_bounds_buffer;
    luisa::shared_ptr<Buffer<BVHPrimitive>> _bvh_primitive_buffer;
    luisa::shared_ptr<Buffer<QBVHNode>> _quantized_bvh_node_buffer;
    luisa::shared_ptr<Buffer<uint>> _bvh_leaf_lut_buffer;
    uint _light_handle_buffer_id{0u};
    uint _tag_lut_buffer_id{0u};
    float _env_prob;

    [[nodiscard]] std::pair<Float3, Float3> _world_bounds() const noexcept {
        return std::make_pair((*_world_bounds_buffer)->read(0u), (*_world_bounds_buffer)->read(1u));
    }
    [[nodiscard]] static Float _evaluate_node(Expr<QBVHNode> node, Expr<float3> p_from, Expr<float3> world_min, Expr<float3> world_max) noexcept {
        auto constexpr QUANTIZED_COORDINATE_BITS = 10u;
        auto constexpr QUANTIZED_COORDINATE_MASK = (1u << QUANTIZED_COORDINATE_BITS) - 1u;
        auto constexpr MAX_COORDINATE_VALUE = static_cast<float>(QUANTIZED_COORDINATE_MASK);
        auto decode_coordinates = [&](auto x) noexcept {
            auto p = make_uint3(x, x >> QUANTIZED_COORDINATE_BITS, x >> (2u * QUANTIZED_COORDINATE_BITS)) & QUANTIZED_COORDINATE_MASK;
            return make_float3(p) / MAX_COORDINATE_VALUE * (world_max - world_min) + world_min;
        };
        auto p_min = decode_coordinates(node.quantized_p_min);
        auto p_max = decode_coordinates(node.quantized_p_max);
        auto centroid = (p_min + p_max) * .5f;
        return node.power / (distance_squared(centroid, p_from) + 1e-6f);
    }
    [[nodiscard]] static Float _evaluate_primitive(Expr<BVHPrimitive> primitive, Expr<float3> p_from) noexcept {
        auto p_min = make_float3(primitive.x_min, primitive.y_min, primitive.z_min);
        auto p_max = make_float3(primitive.x_max, primitive.y_max, primitive.z_max);
        auto centroid = (p_min + p_max) * .5f;
        return primitive.power / (distance_squared(centroid, p_from) + 1e-6f);
    }

public:
    BVHLightSamplerInstance(const BVHLightSampler *sampler, Pipeline &pipeline, CommandBuffer &command_buffer)
        : LightSampler::Instance{pipeline, sampler} {
        if (!pipeline.lights().empty()) {
            auto n = static_cast<uint>(pipeline.geometry()->light_instances().size());
            auto m = static_cast<uint>(pipeline.geometry()->instances().size());
            auto [light_handle_buffer_view, light_handle_buffer_id] = pipeline.bindless_arena_buffer<Light::Handle>(n);
            _light_handle_buffer_id = light_handle_buffer_id;
            auto [tag_lut_buffer_view, tag_lut_buffer_id] = pipeline.bindless_arena_buffer<uint>(m);
            _tag_lut_buffer_id = tag_lut_buffer_id;
            command_buffer << light_handle_buffer_view.copy_from(pipeline.geometry()->light_instances().data()) << commit();
            luisa::vector<uint> tag_lut(m);
            for (auto i = 0u; i < n; i++) {
                auto const &handle = pipeline.geometry()->light_instances()[i];
                tag_lut[handle.instance_id] = i;
            }
            command_buffer << tag_lut_buffer_view.copy_from(tag_lut.data()) << commit();
            _world_bounds_buffer = luisa::make_shared<Buffer<float3>>(pipeline.device().create_buffer<float3>(2u));
            _bvh_primitive_buffer = luisa::make_shared<Buffer<BVHPrimitive>>(pipeline.device().create_buffer<BVHPrimitive>(n));
            _quantized_bvh_node_buffer = luisa::make_shared<Buffer<QBVHNode>>(pipeline.device().create_buffer<QBVHNode>(2u * n - 1u));
            _bvh_leaf_lut_buffer = luisa::make_shared<Buffer<uint>>(pipeline.device().create_buffer<uint>(n));
            _clear_primitive_data = luisa::make_shared<Shader1D<>>(pipeline.device().compile<1>([&]() noexcept {
                set_block_size(256u);
                auto n = static_cast<uint>(pipeline.geometry()->light_instances().size());
                auto i = dispatch_id().x;
                Var<BVHPrimitive> primitive;
                primitive.x_min = std::numeric_limits<float>::max();
                primitive.y_min = std::numeric_limits<float>::max();
                primitive.z_min = std::numeric_limits<float>::max();
                primitive.x_max = std::numeric_limits<float>::lowest();
                primitive.y_max = std::numeric_limits<float>::lowest();
                primitive.z_max = std::numeric_limits<float>::lowest();
                primitive.power = 0.f;
                primitive.tag = 0xffffffffu;
                $while(i < n) {
                    (*_bvh_primitive_buffer)->write(i, primitive);
                    i += dispatch_size().x;
                };
            }));
            _update_primitive_data = luisa::make_shared<Shader1D<float>>(pipeline.device().compile<1>([&](Float time) noexcept {
                set_block_size(256u);
                auto n = static_cast<uint>(pipeline.geometry()->light_instances().size());
                auto tag = def(0u);
                $while(tag < n) {
                    auto handle = pipeline.buffer<Light::Handle>(_light_handle_buffer_id).read(tag);
                    auto light_inst = pipeline.geometry()->instance(handle.instance_id);
                    auto object_to_world = pipeline.geometry()->instance_to_world(handle.instance_id);
                    auto m = make_float3x3(object_to_world);
                    auto t = make_float3(object_to_world[3]);
                    auto power = def(0.f);
                    auto p_min = def(make_float3(std::numeric_limits<float>::max()));
                    auto p_max = def(make_float3(std::numeric_limits<float>::lowest()));
                    auto primitive_id = dispatch_id().x;
                    $while(primitive_id < light_inst.triangle_count()) {
                        auto triangle = pipeline.geometry()->triangle(light_inst, primitive_id);
                        auto v_buffer = light_inst.vertex_buffer_id();
                        auto v0 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i0);
                        auto v1 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i1);
                        auto v2 = pipeline.buffer<Vertex>(v_buffer).read(triangle.i2);
                        auto p0 = m * v0->position() + t;
                        auto p1 = m * v1->position() + t;
                        auto p2 = m * v2->position() + t;
                        p_min = min(p_min, min(p0, min(p1, p2)));
                        p_max = max(p_max, max(p0, max(p1, p2)));
                        auto dp0 = p1 - p0, dp1 = p2 - p0;
                        auto surface_area = .5f * length(cross(dp0, dp1));
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
                        power += surface_area * emission_luminance;
                        primitive_id += dispatch_size().x;
                    };
                    (*_bvh_primitive_buffer)->atomic(tag).x_min.fetch_min(p_min.x);
                    (*_bvh_primitive_buffer)->atomic(tag).y_min.fetch_min(p_min.y);
                    (*_bvh_primitive_buffer)->atomic(tag).z_min.fetch_min(p_min.z);
                    (*_bvh_primitive_buffer)->atomic(tag).x_max.fetch_max(p_max.x);
                    (*_bvh_primitive_buffer)->atomic(tag).y_max.fetch_max(p_max.y);
                    (*_bvh_primitive_buffer)->atomic(tag).z_max.fetch_max(p_max.z);
                    (*_bvh_primitive_buffer)->atomic(tag).power.fetch_add(power);
                    tag += 1u;
                };
            }));
        }
        if (pipeline.environment() != nullptr) {
            _env_prob = pipeline.lights().empty() ? 1.f : std::clamp(sampler->environment_weight(), 0.01f, 0.99f);
        }
    }

    void update(CommandBuffer &command_buffer, float time) noexcept override {
        if (!pipeline().lights().empty()) {
            command_buffer << synchronize();
            auto n = static_cast<uint>(pipeline().geometry()->light_instances().size());
            LUISA_INFO("Reconstructing BVH of {} lights...", n);
            Clock clk;
            luisa::vector<BVHPrimitive> primitive_buffer(n);
            command_buffer << (*_clear_primitive_data)().dispatch(1024u)
                           << (*_update_primitive_data)(time).dispatch(1024u)
                           << _bvh_primitive_buffer->copy_to(primitive_buffer.data())
                           << commit()
                           << synchronize();
            luisa::list<BVHPrimitive> primitive_list;
            for (auto i = 0u; i < n; i++) {
                auto primitive = primitive_buffer[i];
                primitive.tag = i;
                primitive_list.push_back(primitive);
            }
            auto bvh = create_bvh(primitive_list);
            std::copy(primitive_list.begin(), primitive_list.end(), primitive_buffer.begin());
            std::array<float3, 2> world_bounds;
            world_bounds[0] = bvh->p_min;
            world_bounds[1] = bvh->p_max;
            auto quantized_nodes = encode_quantized_bvh(bvh.get(), world_bounds[0], world_bounds[1]);
            luisa::vector<uint> leaf_lut(n);
            for (auto i = 0u; i < quantized_nodes.size(); i++) {
                if (quantized_nodes[i].right_child == 0xffffffffu) {
                    auto first_primitive_offset = quantized_nodes[i].first_primitive_offset;
                    auto last_primitive_offset = first_primitive_offset + quantized_nodes[i].primitive_count;
                    for (auto j = first_primitive_offset; j < last_primitive_offset; j++) {
                        auto tag = primitive_buffer[j].tag;
                        leaf_lut[tag] = i;
                    }
                }
            }
            command_buffer << _quantized_bvh_node_buffer->copy_from(quantized_nodes.data())
                           << _bvh_leaf_lut_buffer->copy_from(leaf_lut.data())
                           << _world_bounds_buffer->copy_from(world_bounds.data())
                           << _bvh_primitive_buffer->copy_from(primitive_buffer.data())
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
                auto current = (*_bvh_leaf_lut_buffer)->read(tag);
                auto current_node = (*_quantized_bvh_node_buffer)->read(current);
                auto [world_min, world_max] = _world_bounds();
                prob = (1.f - _env_prob);
                auto first_primitive_offset = current_node.first_primitive_offset;
                auto last_primitive_offset = first_primitive_offset + current_node.primitive_count;
                auto w1 = def(0.f), w_sum = def(0.f);
                $for(primitive_id, first_primitive_offset, last_primitive_offset) {
                    auto primitive = (*_bvh_primitive_buffer)->read(primitive_id);
                    auto w = _evaluate_primitive(primitive, p_from);
                    $if(primitive.tag == tag) {
                        w1 = w;
                    };
                    w_sum += w;
                };
                prob *= w1 / w_sum;
                $while(current_node.parent != 0xffffffffu) {
                    auto parent = current_node.parent;
                    auto parent_node = (*_quantized_bvh_node_buffer)->read(parent);
                    auto sibling = ite(parent_node.right_child == current, parent + 1u, parent_node.right_child);
                    auto sibling_node = (*_quantized_bvh_node_buffer)->read(sibling);
                    auto w1 = _evaluate_node(current_node, p_from, world_min, world_max);
                    auto w2 = _evaluate_node(sibling_node, p_from, world_min, world_max);
                    prob *= w1 / (w1 + w2);
                    current = parent;
                    current_node = parent_node;
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
        if (_env_prob == 1.f) { return {.tag = LightSampler::selection_environment, .prob = 1.f}; }
        auto uu = (u - _env_prob) / (1.f - _env_prob);
        auto [world_min, world_max] = _world_bounds();
        auto current = def(0u);
        auto current_node = (*_quantized_bvh_node_buffer)->read(current);
        auto prob = def(1.f - _env_prob);
        $while(current_node.right_child != 0xffffffffu) {
            auto left_child = (*_quantized_bvh_node_buffer)->read(current + 1u);
            auto right_child = (*_quantized_bvh_node_buffer)->read(current_node.right_child);
            auto w1 = _evaluate_node(left_child, it_from.p(), world_min, world_max);
            auto w2 = _evaluate_node(right_child, it_from.p(), world_min, world_max);
            auto left_prob = w1 / (w1 + w2);
            $if(uu < left_prob) {
                current = current + 1u;
                current_node = left_child;
                prob *= left_prob;
                uu /= left_prob;
            } $else {
                current = current_node.right_child;
                current_node = right_child;
                prob *= 1.f - left_prob;
                uu = (uu - left_prob) / (1.f - left_prob);
            };
        };
        auto first_primitive_offset = current_node.first_primitive_offset;
        auto last_primitive_offset = first_primitive_offset + current_node.primitive_count;
        auto selected_primitive_id = first_primitive_offset;
        auto w1 = def(0.f), w_sum = def(0.f);
        $for(primitive_id, first_primitive_offset, last_primitive_offset) {
            auto primitive = (*_bvh_primitive_buffer)->read(primitive_id);
            auto w = _evaluate_primitive(primitive, it_from.p());
            w_sum += w;
            auto p = w / w_sum;
            $if(uu < p) {
                w1 = w;
                selected_primitive_id = primitive_id;
                uu /= p;
            };
        };
        auto tag = (*_bvh_primitive_buffer)->read(selected_primitive_id).tag;
        prob *= w1 / w_sum;
        auto is_env = u < _env_prob;
        return {.tag = ite(is_env, LightSampler::selection_environment, tag),
                .prob = ite(is_env, _env_prob, prob)};
    }

    [[nodiscard]] LightSampler::Selection select(
        Expr<float> u, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        LUISA_WARNING_WITH_LOCATION("BVHLightSampler does not support light selection independent of shading point.");
        return {.tag = LightSampler::selection_environment, .prob = 0.f};
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
