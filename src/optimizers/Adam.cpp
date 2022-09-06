//
// Created by Kasumi on 2022/7/7.
//

#include <base/optimizer.h>
#include <base/pipeline.h>
#include <sdl/scene_node_desc.h>

namespace luisa::render {

using namespace luisa::compute;

class Adam final : public Optimizer {

private:
    float _beta1, _beta2;
    float _epsilon;

public:
    Adam(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Optimizer{scene, desc},
          _beta1{std::max(desc->property_float_or_default("beta1", 0.9f), 0.f)},
          _beta2{std::max(desc->property_float_or_default("beta2", 0.999f), 0.f)},
          _epsilon{std::max(desc->property_float_or_default("epsilon", 1e-8f), 1e-40f)} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Optimizer::Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

public:
    [[nodiscard]] auto beta1() const noexcept { return _beta1; }
    [[nodiscard]] auto beta2() const noexcept { return _beta2; }
    [[nodiscard]] auto epsilon() const noexcept { return _epsilon; }
};

class AdamInstance final : public Optimizer::Instance {

private:
    luisa::optional<BufferView<float>> _m, _v;
    luisa::optional<BufferView<float>> _beta_t;

    Shader1D<Buffer<float>, Buffer<float>, Buffer<float>, Buffer<float>, float, float, float> _update_params;

public:
    AdamInstance(Pipeline &pipeline, CommandBuffer &command_buffer, const Adam *optimizer) noexcept
        : Optimizer::Instance{pipeline, command_buffer, optimizer} {

        Kernel1D update_params_kernel = [](BufferFloat m, BufferFloat v, BufferFloat beta_t, BufferFloat gradients,
                                           Float beta1, Float beta2, Float epsilon) noexcept {
            auto index = dispatch_x();
            auto grad = gradients.read(index);
            auto m_tm1 = m.read(index);
            auto v_tm1 = v.read(index);
            auto m_t = beta1 * m_tm1 + (1.f - beta1) * grad;
            auto v_t = beta2 * v_tm1 + (1.f - beta2) * grad * grad;
            auto beta_1_t = beta_t.read(0u) * beta1;
            auto beta_2_t = beta_t.read(1u) * beta2;
            auto m_t_hat = m_t / (1.f - beta_1_t);
            auto v_t_hat = v_t / (1.f - beta_2_t);
            grad = m_t_hat / (sqrt(v_t_hat) + epsilon);
            beta_t.write(0u, beta_1_t);
            beta_t.write(1u, beta_2_t);
            m.write(index, m_t);
            v.write(index, v_t);
            gradients.write(index, grad);
        };
        _update_params = pipeline.device().compile(update_params_kernel);
    }

public:
    void initialize(CommandBuffer &command_buffer, uint length, BufferView<float> xi,
                    BufferView<float> gradients, BufferView<float2> ranges) noexcept override;
    void step(CommandBuffer &command_buffer) noexcept override;
};

void AdamInstance::initialize(CommandBuffer &command_buffer, uint length, BufferView<float> xi,
                              BufferView<float> gradients, BufferView<float2> ranges) noexcept {
    Optimizer::Instance::initialize(command_buffer, length, xi, gradients, ranges);

    command_buffer << synchronize();
    LUISA_INFO("_length = {}, length = {}", _length, length);

    _m.reset();
    _v.reset();
    _beta_t.reset();

    _m.emplace(pipeline().create<Buffer<float>>(std::max(length, 1u))->view());
    _v.emplace(pipeline().create<Buffer<float>>(std::max(length, 1u))->view());
    _beta_t.emplace(pipeline().create<Buffer<float>>(2u)->view());

    command_buffer << _clear_float_buffer(*_m).dispatch(length)
                   << _clear_float_buffer(*_v).dispatch(length);
}

void AdamInstance::step(CommandBuffer &command_buffer) noexcept {
    LUISA_ASSERT(_length != -1u, "Optimizer is not initialized.");
    auto node_exact = node<Adam>();
    command_buffer << _update_params(*_m, *_v, *_beta_t, *_gradients,
                                     node_exact->beta1(), node_exact->beta2(),
                                     node_exact->epsilon())
                          .dispatch(_length);
    clamp_range(command_buffer);
}

luisa::unique_ptr<Optimizer::Instance> Adam::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<AdamInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Adam)