//
// Created by Mike Smith on 2022/10/13.
//

#include <core/mathematics.h>
#include <core/thread_pool.h>
#include <util/imageio.h>
#include <textures/sky_precompute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::render;

int main() {
    NishitaSkyData data{.sun_elevation = radians(23.4f),
                        .sun_angle = radians(.545f),
                        .altitude = 670.f,
                        .air_density = 1.748f,
                        .dust_density = 7.f,
                        .ozone_density = 2.783f};
    static constexpr auto resolution = make_uint2(2048u);
    luisa::vector<float4> image(resolution.x * resolution.y);
    ThreadPool::global().parallel(resolution.y / 16u, [&](uint32_t y) noexcept {
        SKY_nishita_skymodel_precompute_texture(
            data, image.data(), resolution, make_uint2(y * 16u, (y + 1u) * 16u));
    });
    auto sun = SKY_nishita_skymodel_precompute_sun(data);
    LUISA_INFO("Sun: ({}, {}, {}) -> ({}, {}, {})",
               sun.bottom.x, sun.bottom.y, sun.bottom.z,
               sun.top.x, sun.top.y, sun.top.z);
    ThreadPool::global().synchronize();
    save_image("sky_precompute_test.exr",
               reinterpret_cast<const float *>(image.data()),
               resolution);
}
