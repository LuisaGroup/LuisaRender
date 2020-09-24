////
//// Created by Mike Smith on 2020/9/24.
////
//
//#include <algorithm>
//
//namespace luisa {
//
//struct LinearTRSKeyFrame {
//    float time_point;
//    std::shared_ptr<TRSTransform> transform;
//};
//
//class LinearTRSAnimation : public Transform {
//
//private:
//    std::vector<LinearTRSKeyFrame> _key_frames;
//
//public:
//    LinearTRSAnimation(Device *device, const ParameterSet &parameter_set);
//    [[nodiscard]] bool is_static() const noexcept override { return false; }
//    [[nodiscard]] float4x4 dynamic_matrix(float time) const override;
//};
//
//LinearTRSAnimation::LinearTRSAnimation(Device *device, const ParameterSet &parameter_set)
//    : Transform{device, parameter_set} {
//
//    auto time_points = parameter_set["time_points"].parse_float_list();
//    auto transforms = parameter_set["transforms"].parse_reference_list<Transform>();
//
//    LUISA_EXCEPTION_IF(time_points.size() < 2ul || transforms.size() < 2u, "No enough time points and transforms given");
//    LUISA_WARNING_IF(time_points.size() != transforms.size(), "Numbers of time points and transforms mismatch, discarding redundant ones");
//
//    auto key_frame_count = std::max(time_points.size(), transforms.size());
//    time_points.resize(key_frame_count);
//    transforms.resize(key_frame_count);
//
//    for (auto i = 0ul; i < key_frame_count; i++) {
//        auto trs_transform = std::dynamic_pointer_cast<TRSTransform>(transforms[i]);
//        LUISA_EXCEPTION_IF(trs_transform == nullptr, "Only TRSTransform supported");
//        _key_frames.emplace_back(LinearTRSKeyFrame{time_points[i], trs_transform});
//    }
//
//    std::sort(_key_frames.begin(), _key_frames.end(), [](const auto &lhs, const auto &rhs) noexcept {
//        return lhs.time_point < rhs.time_point;
//    });
//
//    auto prev = INFINITY;
//    for (auto &&f : _key_frames) {
//        LUISA_EXCEPTION_IF(f.time_point == prev, "Duplicated time point: ", f.time_point);
//    }
//}
//
//float4x4 LinearTRSAnimation::dynamic_matrix(float time) const {
//
//    LUISA_EXCEPTION_IF(time < _key_frames.front().time_point || time > _key_frames.back().time_point, "Time point not in range: ", time);
//
//    if (time == _key_frames.front().time_point) { return _key_frames.front().transform->static_matrix(); }
//    if (time == _key_frames.back().time_point) { return _key_frames.back().transform->static_matrix(); }
//    auto iter = std::lower_bound(_key_frames.cbegin(), _key_frames.cend(), time, [](const auto &lhs, float rhs) noexcept { return lhs.time_point < rhs; });
//    auto &&next = *iter;
//    auto &&prev = *(--iter);
//    auto alpha = (time - prev.time_point) / (next.time_point - prev.time_point);
//
//    auto t = math::lerp(prev.transform->translation(), next.transform->translation(), alpha);
//    auto r = math::lerp(prev.transform->rotation(), next.transform->rotation(), alpha);
//    auto s = math::lerp(prev.transform->scaling(), next.transform->scaling(), alpha);
//
//    return math::translation(t) * math::rotation(make_float3(r), r.a) * math::scaling(s);
//}
//
//}
//
//LUISA_EXPORT_PLUGIN_CREATOR(luisa::LinearTRSAnimation)
