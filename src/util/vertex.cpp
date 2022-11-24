//
// Created by Mike Smith on 2022/11/24.
//

#include <util/vertex.h>

namespace luisa::render {

[[nodiscard]] inline auto fallback_tangent(float3 n) noexcept {
    auto b = abs(n.x) > abs(n.z) ?
                 make_float3(-n.y, n.x, 0.f) :
                 make_float3(0.f, -n.z, n.y);
    return cross(b, n);
}

float3 compute_tangent(float3 p0, float3 p1, float3 p2,
                       float2 uv0, float2 uv1, float2 uv2) noexcept {
    auto difference_of_products = [](auto a, auto b, auto c, auto d) noexcept {
        auto cd = c * d;
        auto differenceOfProducts = a * b - cd;
        auto error = -c * d + cd;
        return differenceOfProducts + error;
    };
    auto length_squared = [](auto v) noexcept { return dot(v, v); };
    auto duv02 = uv0 - uv2;
    auto duv12 = uv1 - uv2;
    auto dp02 = p0 - p2;
    auto dp12 = p1 - p2;
    auto det = difference_of_products(duv02.x, duv12.y, duv02.y, duv12.x);
    auto dpdu = make_float3();
    auto dpdv = make_float3();
    auto degenerate_uv = abs(det) < 1e-8f;
    if (!degenerate_uv) {
        // Compute triangle $\dpdu$ and $\dpdv$ via matrix inversion
        auto invdet = 1.f / det;
        dpdu = difference_of_products(duv12.y, dp02, duv02.y, dp12) * invdet;
        dpdv = difference_of_products(duv02.x, dp12, duv12.x, dp02) * invdet;
    }
    // Handle degenerate triangle $(u,v)$ parameterization or partial derivatives
    if (degenerate_uv || length_squared(cross(dpdu, dpdv)) == 0.f) {
        dpdu = fallback_tangent(cross(p2 - p0, p1 - p0));
    };
    return dpdu;
}

void compute_tangents(luisa::span<Vertex> vertices,
                      luisa::span<const compute::Triangle> triangles,
                      bool area_weighted) noexcept {
    luisa::vector<float3> tangents(vertices.size(), make_float3());
    for (auto t : triangles) {
        auto v0 = vertices[t.i0];
        auto v1 = vertices[t.i1];
        auto v2 = vertices[t.i2];
        auto p0 = v0.position();
        auto p1 = v1.position();
        auto p2 = v2.position();
        auto uv0 = v0.uv();
        auto uv1 = v1.uv();
        auto uv2 = v2.uv();
        auto weight = area_weighted ? length(cross(p1 - p0, p2 - p0)) : 1.f;
        auto tangent = weight * compute_tangent(p0, p1, p2, uv0, uv1, uv2);
        tangents[t.i0] += tangent;
        tangents[t.i1] += tangent;
        tangents[t.i2] += tangent;
    }
    for (auto i = 0u; i < vertices.size(); i++) {
        auto n = vertices[i].normal();
        auto t = tangents[i];
        if (dot(t, t) == 0.f) { t = fallback_tangent(-n); }
        vertices[i].s = oct_encode(normalize(t));
    }
}

}// namespace luisa::render
