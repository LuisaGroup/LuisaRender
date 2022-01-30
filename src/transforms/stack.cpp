//
// Created by Mike Smith on 2022/1/15.
//

#include <base/transform.h>
#include <base/scene.h>

namespace luisa::render {

class TransformStack final : public Transform {

private:
    luisa::vector<const Transform *> _transforms;
    mutable float4x4 _matrix_cache;
    mutable float _time_cache;
    mutable spin_mutex _mutex;
    bool _is_static;
    bool _is_identity;

public:
    TransformStack(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Transform{scene, desc},
          _matrix_cache{make_float4x4(1.0f)}, _time_cache{0.0f},
          _is_static{true}, _is_identity{true} {
        auto children = desc->property_node_list_or_default("transforms");
        luisa::vector<const Transform *> transforms(children.size());
        for (auto i = 0u; i < children.size(); i++) {
            auto t = scene->load_transform(children[i]);
            transforms[i] = t;
            _is_static &= t->is_static();
            _is_identity &= t->is_identity();
            _matrix_cache = t->matrix(_time_cache) * _matrix_cache;
        }
        if (!_is_static) { _transforms = std::move(transforms); }
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_static() const noexcept override { return _is_static; }
    [[nodiscard]] bool is_identity() const noexcept override { return _is_identity; }
    [[nodiscard]] float4x4 matrix(float time) const noexcept override {
        if (_is_static) { return _matrix_cache; }
        if (_transforms.size() < 4u) {
            auto m = make_float4x4(1.0f);
            for (auto t : _transforms) {
                m = t->matrix(time) * m;
            }
            return m;
        }
        std::scoped_lock lock{_mutex};
        if (time != _time_cache) {
            _time_cache = time;
            _matrix_cache = make_float4x4(1.0f);
            for (auto t : _transforms) {
                auto m = t->matrix(_time_cache);
                _matrix_cache = m * _matrix_cache;
            }
        }
        return _matrix_cache;
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::TransformStack)
