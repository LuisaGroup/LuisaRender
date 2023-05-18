//
// Created by Mike Smith on 2022/11/8.
//

#include <fstream>

#define LUISA_RENDER_PLUGIN_NAME "sphere"
#include <shapes/sphere.cpp>
#include <core/clock.h>
#include <util/thread_pool.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

void dump_obj(MeshView m, uint level) noexcept {
    auto path = format("sphere-{}.obj", level);
    std::ofstream out{path.c_str()};
    auto dump_vertex = [&out](auto v, auto t) noexcept {
        if (std::same_as<decltype(v), float2>) {
            out << luisa::format("{} {} {}\n", t, v[0], v[1]);
        } else {
            out << luisa::format("{} {} {} {}\n", t, v[0], v[1], v[2]);
        }
    };
    for (auto v : m.vertices) { dump_vertex(v.position(), "v"); }
    for (auto v : m.vertices) { dump_vertex(v.normal(), "vn"); }
    for (auto v : m.vertices) { dump_vertex(v.uv(), "vt"); }
    for (auto [a, b, c] : m.triangles) {
        out << luisa::format("f {}/{}/{} {}/{}/{} {}/{}/{}\n",
                             a + 1u, a + 1u, a + 1u,
                             b + 1u, b + 1u, b + 1u,
                             c + 1u, c + 1u, c + 1u);
    }
}

int main() {
    static_cast<void>(global_thread_pool());
    for (auto i = 0u; i <= sphere_max_subdivision_level; i++) {
        Clock clk;
        auto future = SphereGeometry::create(i);
        auto m = future.get().mesh();
        LUISA_INFO("Computed sphere at subdivision level {} "
                   "with {} vertices and {} triangles in {} ms.",
                   i, m.vertices.size(), m.triangles.size(), clk.toc());
        dump_obj(m, i);
    }
}
