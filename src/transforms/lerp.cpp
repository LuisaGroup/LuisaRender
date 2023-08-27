//
// Created by Mike Smith on 2022/1/15.
//

#include <numeric>

#include <util/xform.h>
#include <base/transform.h>
#include <base/scene.h>

namespace luisa::render {

class LerpTransform final : public Transform {

private:
    luisa::vector<const Transform *> _transforms;
    luisa::vector<float> _time_points;
    mutable float4x4 _matrix_cache;
    mutable float _time_cache;
    mutable uint _upper_index_cache{~0u};
    mutable spin_mutex _mutex;
    mutable DecomposedTransform _t0_cache{};
    mutable DecomposedTransform _t1_cache{};

public:
    LerpTransform(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Transform{scene, desc},
          _matrix_cache{make_float4x4(1.0f)},
          _time_cache{std::numeric_limits<float>::quiet_NaN()} {
        auto nodes = desc->property_node_list("transforms");
        auto times = desc->property_float_list("time_points");
        if (nodes.size() != times.size()) [[unlikely]] {
            LUISA_ERROR(
                "Number of transforms and number "
                "of time points mismatch. [{}]",
                desc->source_location().string());
        }
        if (nodes.empty()) [[unlikely]] {
            LUISA_ERROR(
                "Empty transform list. [{}]",
                desc->source_location().string());
        }
        luisa::vector<uint> indices(times.size());
        std::iota(indices.begin(), indices.end(), 0u);
        std::sort(
            indices.begin(), indices.end(),
            [&times](auto lhs, auto rhs) noexcept {
                return times[lhs] < times[rhs];
            });
        if (auto iter = std::unique(
                indices.begin(), indices.end(),
                [&times](auto lhs, auto rhs) noexcept {
                    return times[lhs] == times[rhs];
                });
            iter != indices.end()) [[unlikely]] {
            LUISA_WARNING(
                "Duplicate time points (count = {}) in "
                "LerpTransform will be removed. [{}]",
                std::distance(iter, indices.end()),
                desc->source_location().string());
            indices.erase(iter, indices.end());
        }
        _time_points.reserve(indices.size());
        _transforms.reserve(indices.size());
        for (auto index : indices) {
            _time_points.emplace_back(times[index]);
            _transforms.emplace_back(
                scene->load_transform(nodes[index]));
        }
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_static() const noexcept override { return false; }
    [[nodiscard]] bool is_identity() const noexcept override { return false; }
    [[nodiscard]] float4x4 matrix(float time) const noexcept override {
        std::scoped_lock lock{_mutex};
        if (time != _time_cache) {
            if (time <= _time_points.front()) {
                _time_cache = _time_points.front();
                _matrix_cache = _transforms.front()->matrix(_time_cache);
            } else if (time >= _time_points.back()) {
                _time_cache = _time_points.back();
                _matrix_cache = _transforms.back()->matrix(_time_cache);
            } else {
                // _time_points.front() < time < _time_points.back()
                _time_cache = time;
                auto upper = std::upper_bound(
                    _time_points.cbegin(), _time_points.cend(), _time_cache);
                if (_time_points.size() < 2u ||
                    upper == _time_points.cend()) [[unlikely]] {
                    LUISA_ERROR_WITH_LOCATION("This is impossible.");
                }
                auto upper_index = upper - _time_points.cbegin();
                if (upper_index != _upper_index_cache) {
                    _upper_index_cache = static_cast<uint>(upper_index);
                    _t0_cache = decompose(
                        _transforms[upper_index - 1u]->matrix(_time_cache));
                    _t1_cache = decompose(
                        _transforms[upper_index]->matrix(_time_cache));
                }
                auto time_lower = _time_points[_upper_index_cache - 1u];
                auto time_upper = _time_points[_upper_index_cache];
                auto t = (_time_cache - time_lower) / (time_upper - time_lower);
                auto S = lerp(_t0_cache.scaling, _t1_cache.scaling, t);
                auto R = slerp(_t0_cache.quaternion, _t1_cache.quaternion, t);
                auto T = lerp(_t0_cache.translation, _t1_cache.translation, t);
                _matrix_cache = translation(T) * rotation(R) * scaling(S);
            }
        }
        return _matrix_cache;
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::LerpTransform)
