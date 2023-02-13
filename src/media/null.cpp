//
// Created by ChenXin on 2023/2/13.
//

#include <base/medium.h>

namespace luisa::render {

class NullMedium : public Medium {

public:
    class NullMediumInstance;

    class NullMediumInstance : public Medium::Instance {

    protected:
        friend class NullMedium;

    public:
        NullMediumInstance(const Pipeline &pipeline, const Medium *medium) noexcept
            : Medium::Instance(pipeline, medium) {}
    };

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override {
        return luisa::make_unique<NullMediumInstance>(pipeline, this);
    }

public:
    NullMedium(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Medium{scene, desc} {}
    [[nodiscard]] bool is_null() const noexcept override { return true; }

};

}// namespace luisa::render