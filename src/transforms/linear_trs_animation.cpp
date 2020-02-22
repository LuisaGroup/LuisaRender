//
// Created by Mike Smith on 2020/2/20.
//

#include <algorithm>
#include "linear_trs_animation.h"

namespace luisa {

LUISA_REGISTER_NODE_CREATOR("LinearTRSAnimation", LinearTRSAnimation)

LinearTRSAnimation::LinearTRSAnimation(Device *device, const ParameterSet &parameter_set)
    : Transform{device, parameter_set} {
    
    auto time_points = parameter_set["time_points"].parse_float_list();
    auto transforms = parameter_set["transforms"].parse_reference_list<Transform>();
    
    LUISA_ERROR_IF(time_points.size() < 2ul || transforms.size() < 2u, "no enough time points and transforms given");
    LUISA_WARNING_IF(time_points.size() != transforms.size(), "numbers of time points and transforms mismatch, discarding redundant ones");
    
    auto key_frame_count = std::max(time_points.size(), transforms.size());
    time_points.resize(key_frame_count);
    transforms.resize(key_frame_count);
    
    for (auto i = 0ul; i < key_frame_count; i++) {
        auto trs_transform = std::dynamic_pointer_cast<TRSTransform>(transforms[i]);
        LUISA_ERROR_IF(trs_transform == nullptr, "only TRSTransform supported");
        _key_frames.emplace_back(LinearTRSKeyFrame{time_points[i], trs_transform});
    }
    
    std::sort(_key_frames.begin(), _key_frames.end(), [](const auto &lhs, const auto &rhs) noexcept {
        return lhs.time_point < rhs.time_point;
    });
    
    auto prev = INFINITY;
    for (auto &&f : _key_frames) {
        LUISA_ERROR_IF(f.time_point == prev, "duplicated time point: ", f.time_point);
    }
}

bool LinearTRSAnimation::is_static() const noexcept {
    return false;
}

float4x4 LinearTRSAnimation::dynamic_matrix(float time) const {
    
    LUISA_ERROR_IF(time < _key_frames.front().time_point || time > _key_frames.back().time_point, "time point not in range: ", time);
    
    if (time == _key_frames.front().time_point) { return _key_frames.front().transform->static_matrix(); }
    if (time == _key_frames.back().time_point) { return _key_frames.back().transform->static_matrix(); }
    auto iter = std::lower_bound(_key_frames.cbegin(), _key_frames.cend(), time, [](const auto &lhs, float rhs) noexcept { return lhs.time_point < rhs; });
    auto &&next = *iter;
    auto &&prev = *(--iter);
    auto alpha = (time - prev.time_point) / (next.time_point - prev.time_point);
    
    auto t = lerp(prev.transform->translation(), next.transform->translation(), alpha);
    auto r = lerp(prev.transform->rotation(), next.transform->rotation(), alpha);
    auto s = lerp(prev.transform->scaling(), next.transform->scaling(), alpha);
    
    return translation(t) * rotation(make_float3(r), r.a) * scaling(s);
}

}
