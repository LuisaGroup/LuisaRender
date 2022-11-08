//
// Created by Mike Smith on 2022/11/8.
//

#define LUISA_RENDER_PLUGIN_NAME "sphere"
#include <shapes/sphere.cpp>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

void dump_obj(const SphereGeometry &g, uint level) noexcept {
    std::ofstream out{luisa::format("sphere-{}.obj", level)};
    auto dump_vertex = [&out](auto v, auto t) noexcept {
        if (v.size() == 2u) {
            out << luisa::format("{} {} {}\n", t, v[0], v[1]);
        } else {
            out << luisa::format("{} {} {} {}\n", t, v[0], v[1], v[2]);
        }
    };
    for (auto v : g.vertices()) { dump_vertex(v.compressed_p, "v"); }
    for (auto v : g.vertices()) { dump_vertex(v.compressed_n, "vn"); }
    for (auto v : g.vertices()) { dump_vertex(v.compressed_uv, "vt"); }
    for (auto [a, b, c] : g.triangles()) {
        out << luisa::format("f {}/{}/{} {}/{}/{} {}/{}/{}\n",
                             a + 1u, a + 1u, a + 1u,
                             b + 1u, b + 1u, b + 1u,
                             c + 1u, c + 1u, c + 1u);
    }
}

int main() {
    static_cast<void>(ThreadPool::global());
    for (auto i = 0u; i <= sphere_max_subdivision_level; i++) {
        Clock clk;
        auto future = SphereGeometry::create(i);
        auto &&g = future.get();
        LUISA_INFO("Computed sphere at subdivision level {} with {} vertices and {} triangles in {} ms.",
                   i, g.vertices().size(), g.triangles().size(), clk.toc());
        dump_obj(g, i);
    }
}
