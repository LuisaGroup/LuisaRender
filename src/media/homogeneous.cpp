//
// Created by ChenXin on 2023/2/13.
//

#include <base/medium.h>

namespace luisa::render {

class HomogeneousMedium : public Medium {

protected:
    float3 _sigma_a;
    float3 _sigma_s;
    float3 _sigma_t;

public:
    class HomogeneousMediumInstance;

    class HomogeneousMediumInstance : public Medium::Instance {

    public:
        [[nodiscard]] float3 sigma_a() const noexcept { return node<HomogeneousMedium>()->_sigma_a; }
        [[nodiscard]] float3 sigma_s() const noexcept { return node<HomogeneousMedium>()->_sigma_s; }
        [[nodiscard]] float3 sigma_t() const noexcept { return node<HomogeneousMedium>()->_sigma_t; }

    protected:
        friend class HomogeneousMedium;

    public:
        HomogeneousMediumInstance(const Pipeline &pipeline, const Medium *medium) noexcept
            : Medium::Instance(pipeline, medium) {}
    };

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<HomogeneousMediumInstance>(pipeline, this);
    }

public:
    HomogeneousMedium(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Medium{scene, desc} {
        _sigma_a = desc->property_float3_or_default("sigma_a", make_float3(0.0f));
        _sigma_s = desc->property_float3_or_default("sigma_s", make_float3(0.0f));
        _sigma_t = _sigma_a + _sigma_s;
    }

};

}// namespace luisa::render