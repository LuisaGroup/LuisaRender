//
// Created by Mike Smith on 2022/4/7.
//

#include <util/rng.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

class DemoSDFTexture final : public Texture {

public:
    DemoSDFTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Texture{scene, desc} { disable_gradients(); }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] bool is_constant() const noexcept override { return false; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint channels() const noexcept override { return 3u; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

using namespace luisa::compute;

class DemoSDFTextureInstance final : public Texture::Instance {

private:

public:
    DemoSDFTextureInstance(Pipeline &p, const Texture *t) noexcept
        : Texture::Instance{p, t} {}

    [[nodiscard]] Float4 evaluate(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override {

        static constexpr auto max_ray_depth = 6;
        static constexpr auto eps = 1e-4f;
        static constexpr auto inf = 1e10f;
        static constexpr auto fov = radians(20.f);
        static constexpr auto dist_limit = 100.0f;
        static constexpr auto camera_pos = make_float3(0.0f, 0.32f, 3.7f);
        static constexpr auto light_pos = make_float3(-1.5f, 0.6f, 0.3f);
        static constexpr auto light_normal = make_float3(1.0f, 0.0f, 0.0f);
        static constexpr auto light_radius = 2.0f;
        constexpr auto resolution = make_float2(1.f);

        Callable intersect_light = [](Float3 pos, Float3 d) noexcept {
            auto cos_w = dot(-d, light_normal);
            auto dist = dot(d, light_pos - pos);
            auto D = dist / cos_w;
            auto dist_to_center = distance_squared(light_pos, pos + D * d);
            auto valid = cos_w > 0.0f & dist > 0.0f & dist_to_center < light_radius * light_radius;
            return ite(valid, D, inf);
        };

        Callable tea = [](UInt v0, UInt v1) noexcept {
            Var s0 = 0u;
            for (auto n = 0u; n < 4u; n++) {
                s0 += 0x9e3779b9u;
                v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
                v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
            }
            return v0;
        };

        Callable rand = [](UInt &state) noexcept {
            constexpr auto lcg_a = 1664525u;
            constexpr auto lcg_c = 1013904223u;
            state = lcg_a * state + lcg_c;
            return cast<float>(state & 0x00ffffffu) * (1.0f / static_cast<float>(0x01000000u));
        };

        Callable out_dir = [&rand](Float3 n, UInt &seed) noexcept {
            auto u = ite(
                abs(n.y) < 1.0f - eps,
                normalize(cross(n, make_float3(0.0f, 1.0f, 0.0f))),
                make_float3(1.f, 0.f, 0.f));
            auto v = cross(n, u);
            auto phi = 2.0f * pi * rand(seed);
            auto ay = sqrt(rand(seed));
            auto ax = sqrt(1.0f - ay * ay);
            return ax * (cos(phi) * u + sin(phi) * v) + ay * n;
        };

        Callable make_nested = [](Float f) noexcept {
            static constexpr auto freq = 40.0f;
            f *= freq;
            f = ite(f < 0.f, ite(cast<int>(f) % 2 == 0, 1.f - fract(f), fract(f)), f);
            return (f - 0.2f) * (1.0f / freq);
        };

        Callable sdf = [&make_nested](Float3 o) noexcept {
            auto wall = min(o.y + 0.1f, o.z + 0.4f);
            auto sphere = distance(o, make_float3(0.0f, 0.35f, 0.0f)) - 0.36f;
            auto q = abs(o - make_float3(0.8f, 0.3f, 0.0f)) - 0.3f;
            auto box = length(max(q, 0.0f)) + min(max(max(q.x, q.y), q.z), 0.0f);
            auto O = o - make_float3(-0.8f, 0.3f, 0.0f);
            auto d = make_float2(length(make_float2(O.x, O.z)) - 0.3f, abs(O.y) - 0.3f);
            auto cylinder = min(max(d.x, d.y), 0.0f) + length(max(d, 0.0f));
            auto geometry = make_nested(min(min(sphere, box), cylinder));
            auto g = max(geometry, -(0.32f - (o.y * 0.6f + o.z * 0.8f)));
            return min(wall, g);
        };

        Callable ray_march = [&sdf](Float3 p, Float3 d) noexcept {
            auto dist = def(0.0f);
            $for(j, 100) {
                auto s = sdf(p + dist * d);
                $if(s <= 1e-6f | dist >= inf) { $break; };
                dist += s;
            };
            return min(dist, inf);
        };

        Callable sdf_normal = [&sdf](Float3 p) noexcept {
            static constexpr auto d = 1e-3f;
            auto n = def(make_float3());
            auto sdf_center = sdf(p);
            for (auto i = 0; i < 3; i++) {
                auto inc = p;
                inc[i] += d;
                n[i] = (1.0f / d) * (sdf(inc) - sdf_center);
            }
            return normalize(n);
        };

        Callable next_hit = [&ray_march, &sdf_normal](Float &closest, Float3 &normal, Float3 &c, Float3 pos, Float3 d) noexcept {
            closest = inf;
            normal = make_float3();
            c = make_float3();
            auto ray_march_dist = ray_march(pos, d);
            $if(ray_march_dist < min(dist_limit, closest)) {
                closest = ray_march_dist;
                auto hit_pos = pos + d * closest;
                normal = sdf_normal(hit_pos);
                auto t = cast<int>((hit_pos.x + 10.0f) * 1.1f + 0.5f) % 3;
                c = make_float3(0.4f) + make_float3(0.3f, 0.2f, 0.3f) * ite(t == make_int3(0, 1, 2), 1.0f, 0.0f);
            };
        };

        Callable render = [&](UInt seed, Float2 uv_in) noexcept {
            auto aspect_ratio = resolution.x / resolution.y;
            auto pos = def(camera_pos);
            auto uv = resolution * uv_in;
            auto d = make_float3(2.0f * fov * uv / resolution.y - fov * make_float2(aspect_ratio, 1.0f) - 1e-5f, -1.0f);
            d = normalize(d);
            auto throughput = def(make_float3(1.0f, 1.0f, 1.0f));
            auto hit_light = def(0.0f);
            $for(depth, max_ray_depth) {
                auto closest = def(0.0f);
                auto normal = def(make_float3());
                auto c = def(make_float3());
                next_hit(closest, normal, c, pos, d);
                auto dist_to_light = intersect_light(pos, d);
                $if(dist_to_light < closest) {
                    hit_light = 1.0f;
                    $break;
                };
                $if(length_squared(normal) == 0.0f) { $break; };
                auto hit_pos = pos + closest * d;
                d = out_dir(normal, seed);
                pos = hit_pos + 1e-4f * d;
                throughput *= c;
            };
            return throughput.zyx() * hit_light;
        };
        auto uv = it.uv();
        auto seed = xxhash32(as<uint3>(it.p()));
        auto color = render(seed, uv);
        return make_float4(color, 1.0f);
    }

    void backward(const Interaction &, const SampledWavelengths &swl,
                  Expr<float>, Expr<float4> grad) const noexcept override {
        if (node()->requires_gradients()) {
            LUISA_ERROR_WITH_LOCATION("Not supported.");
        }
    }
};

luisa::unique_ptr<Texture::Instance> DemoSDFTexture::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<DemoSDFTextureInstance>(pipeline, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DemoSDFTexture)
