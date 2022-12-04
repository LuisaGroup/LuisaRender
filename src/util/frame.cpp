//
// Created by Mike Smith on 2022/1/13.
//

#include <util/frame.h>

namespace luisa::render {

inline Frame::Frame(Float3 tangent, Float3 bitangent, Float3 normal) noexcept
    : _u{std::move(tangent)},
      _v{std::move(bitangent)},
      _n{std::move(normal)} {}

Frame::Frame() noexcept
    : _u{1.0f, 0.0f, 0.0f},
      _v{0.0f, 1.0f, 0.0f},
      _n{0.0f, 0.0f, 1.0f} {}

[[nodiscard]] inline auto fallback_tangent(Expr<float3> n) noexcept {
    return normalize(ite(
        abs(n.x) > abs(n.z),
        make_float3(n.z, 0.f, -n.x),
        make_float3(0.f, n.z, -n.y)));
}

Frame Frame::make(Expr<float3> normal) noexcept {
    auto tangent = normalize(ite(
        abs(normal.x) > abs(normal.z),
        make_float3(normal.z, 0.0f, -normal.x),
        make_float3(0.0f, normal.z, -normal.y)));
    auto bitangent = normalize(cross(normal, tangent));
    return Frame{std::move(tangent), std::move(bitangent), normal};
}

Frame Frame::make(Expr<float3> normal, Expr<float3> tangent_in) noexcept {
    auto tangent = ite(all(tangent_in == 0.f), fallback_tangent(normal), tangent_in);
    auto bitangent = normalize(cross(normal, tangent));
    return Frame{normalize(cross(bitangent, normal)), std::move(bitangent), normal};
}

Float3 Frame::local_to_world(Expr<float3> d) const noexcept {
    return d.x * _u + d.y * _v + d.z * _n;
}

Float3 Frame::world_to_local(Expr<float3> d) const noexcept {
    using namespace luisa::compute;
    return make_float3(dot(d, _u), dot(d, _v), dot(d, _n));
}

void Frame::flip() noexcept {
    _n = -_n;
    _v = -_v;
}

}// namespace luisa::render
