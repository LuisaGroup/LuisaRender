//
// Created by Mike Smith on 2022/1/15.
//

#include <numbers>

#include <util/sampling.h>
#include <util/imageio.h>
#include <base/interaction.h>
#include <base/environment.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

class Spherical final : public Environment {

public:
    static constexpr auto sample_map_size = make_uint2(2048u, 1024u);

private:
    const Texture *_emission;
    float _scale;
    bool _compensate_mis;

public:
    Spherical(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Environment{scene, desc},
          _emission{scene->load_texture(desc->property_node("emission"))},
          _scale{std::max(desc->property_float_or_default("scale", 1.0f), 0.0f)},
          _compensate_mis{desc->property_bool_or_default("compensate_mis", true)} {}
    [[nodiscard]] auto scale() const noexcept { return _scale; }
    [[nodiscard]] auto compensate_mis() const noexcept { return _compensate_mis; }
    [[nodiscard]] auto emission() const noexcept { return _emission; }
    [[nodiscard]] bool is_black() const noexcept override { return _scale == 0.0f || _emission->is_black(); }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;

public:
    [[nodiscard]] static auto uv_to_direction(Expr<float2> uv) noexcept {
        auto phi = 2.f * pi * (1.f - uv.x);
        auto theta = pi * uv.y;
        auto y = cos(theta);
        auto sin_theta = sin(theta);
        auto x = sin(phi) * sin_theta;
        auto z = cos(phi) * sin_theta;
        return std::make_tuple(theta, phi, normalize(make_float3(x, y, z)));
    }
    [[nodiscard]] static auto direction_to_uv(Expr<float3> w) noexcept {
        auto theta = acos(w.y);
        auto phi = atan2(w.x, w.z);
        auto u = 1.f - 0.5f * inv_pi * phi;
        auto v = theta * inv_pi;
        return std::make_tuple(theta, phi, fract(make_float2(u, v)));
    }
};

class SphericalInstance final : public Environment::Instance {

private:
    const Texture::Instance *_texture;
    luisa::optional<uint> _alias_buffer_id;
    luisa::optional<uint> _pdf_buffer_id;

private:
    [[nodiscard]] auto _evaluate(Expr<float3> wi_local, Expr<float2> uv,
                                 const SampledWavelengths &swl, Expr<float> time) const noexcept {
        Interaction it{uv};
        auto L = _texture->evaluate_illuminant_spectrum(it, swl, time).value;
        return L * node<Spherical>()->scale();
    }

    [[nodiscard]] static auto _directional_pdf(Expr<float> p, Expr<float> theta) noexcept {
        auto s = sin(theta);
        auto inv_s = ite(s > 0.f, 1.f / s, 0.f);
        return p * inv_s * (.5f * inv_pi * inv_pi);
    }

public:
    SphericalInstance(Pipeline &pipeline, const Environment *env, const Texture::Instance *texture,
                      luisa::optional<uint> alias_buffer_id, luisa::optional<uint> pdf_buffer_id) noexcept
        : Environment::Instance{pipeline, env}, _texture{texture},
          _alias_buffer_id{std::move(alias_buffer_id)},
          _pdf_buffer_id{std::move(pdf_buffer_id)} {}
    [[nodiscard]] Light::Evaluation evaluate(
        Expr<float3> wi, const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto world_to_env = transpose(transform_to_world());
        auto wi_local = normalize(world_to_env * wi);
        auto [theta, phi, uv] = Spherical::direction_to_uv(wi_local);
        auto L = _evaluate(wi_local, uv, swl, time);
        if (_texture->node()->is_constant()) {
            return {.L = L, .pdf = uniform_sphere_pdf()};
        }
        auto size = make_float2(Spherical::sample_map_size);
        auto ix = cast<uint>(clamp(uv.x * size.x, 0.f, size.x - 1.f));
        auto iy = cast<uint>(clamp(uv.y * size.y, 0.f, size.y - 1.f));
        auto pdf_buffer = pipeline().buffer<float>(*_pdf_buffer_id);
        auto pdf = pdf_buffer.read(iy * Spherical::sample_map_size.x + ix);
        return {.L = L, .pdf = _directional_pdf(pdf, theta)};
    }
    [[nodiscard]] Light::Sample sample(
        Expr<float3> p_from, const SampledWavelengths &swl,
        Expr<float> time, Expr<float2> u) const noexcept override {
        auto [wi, Li, pdf] = [&] {
            if (_texture->node()->is_constant()) {
                auto w = sample_uniform_sphere(u);
                auto [theta, phi, uv] = Spherical::direction_to_uv(w);
                auto L = _evaluate(w, uv, swl, time);
                return std::make_tuple(w, L, def(uniform_sphere_pdf()));
            }
            auto alias_buffer = pipeline().buffer<AliasEntry>(*_alias_buffer_id);
            auto [iy, uy] = sample_alias_table(
                alias_buffer, Spherical::sample_map_size.y, u.y);
            auto offset = Spherical::sample_map_size.y +
                          iy * Spherical::sample_map_size.x;
            auto [ix, ux] = sample_alias_table(
                alias_buffer, Spherical::sample_map_size.x, u.x, offset);
            auto uv = make_float2(cast<float>(ix) + ux, cast<float>(iy) + uy) /
                      make_float2(Spherical::sample_map_size);
            auto index = iy * Spherical::sample_map_size.x + ix;
            auto p = pipeline().buffer<float>(*_pdf_buffer_id).read(index);
            auto [theta, phi, w] = Spherical::uv_to_direction(uv);
            auto L = _evaluate(w, uv, swl, time);
            return std::make_tuple(w, L, _directional_pdf(p, theta));
        }();
        return {.eval = {.L = Li, .pdf = pdf},
                .wi = normalize(transform_to_world() * wi),
                .distance = std::numeric_limits<float>::max()};
    }
};

luisa::unique_ptr<Environment::Instance> Spherical::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto texture = pipeline.build_texture(command_buffer, _emission);
    luisa::optional<uint> alias_id;
    luisa::optional<uint> pdf_id;
    if (!_emission->is_constant()) {
        command_buffer << pipeline.bindless_array().update() << commit();
        auto &&device = pipeline.device();
        constexpr auto pixel_count = sample_map_size.x * sample_map_size.y;
        luisa::vector<float> scale_map(pixel_count);
        auto scale_map_device = device.create_buffer<float>(pixel_count);
        Kernel2D generate_weight_map_kernel = [&] {
            auto pixel = dispatch_id().xy();
            auto center = make_float2(pixel) + .5f;
            auto sum_weight = def(0.f);
            auto sum_scale = def(0.f);
            constexpr auto filter_radius = 1.f;
            constexpr auto filter_step = .125f;
            auto n = static_cast<int>(std::ceil(filter_radius / filter_step));
            // kind of brute-force but it's only done once
            $for(dy, -n, n + 1) {
                $for(dx, -n, n + 1) {
                    auto offset = make_float2(make_int2(dx, dy)) * filter_step;
                    auto uv = (center + offset) / make_float2(sample_map_size);
                    auto [theta, phi, w] = Spherical::uv_to_direction(uv);
                    auto it = Interaction{uv};
                    auto scale = texture->evaluate_illuminant_spectrum(it, pipeline.spectrum()->sample(0.5f), 0.f).strength;
                    auto sin_theta = sin(uv.y * pi);
                    auto weight = exp(-4.f * length_squared(offset));// gaussian kernel with an approximate radius of 1
                    auto value = scale * weight * sin_theta;
                    sum_weight += weight;
                    sum_scale += value;
                };
            };
            auto pixel_id = pixel.y * sample_map_size.x + pixel.x;
            scale_map_device.write(pixel_id, sum_scale / sum_weight);
        };
        auto generate_weight_map = device.compile(generate_weight_map_kernel);
        Clock clk;
        command_buffer << generate_weight_map().dispatch(sample_map_size)
                       << scale_map_device.copy_to(scale_map.data())
                       << synchronize();
        LUISA_INFO_WITH_LOCATION(
            "Spherical::build: Generated weight map in {} ms.", clk.toc());
        if (compensate_mis()) {
            auto sum_scale = 0.;
            for (auto s : scale_map) { sum_scale += s; }
            auto average_scale = static_cast<float>(sum_scale / pixel_count);
            for (auto &&s : scale_map) { s = std::max(s - average_scale, 0.f); }
        }
        luisa::vector<float> row_averages(sample_map_size.y);
        luisa::vector<float> pdfs(pixel_count);
        luisa::vector<AliasEntry> aliases(sample_map_size.y + pixel_count);
        // construct conditional alias table
        for (auto i = 0u; i < sample_map_size.y; i++) {
            auto sum = 0.;
            auto values = luisa::span{scale_map}.subspan(
                i * sample_map_size.x, sample_map_size.x);
            for (auto v : values) { sum += v; }
            row_averages[i] = static_cast<float>(sum * (1.0 / sample_map_size.x));
            auto [alias_table, pdf_table] = create_alias_table(values);
            std::copy_n(
                pdf_table.data(), sample_map_size.x,
                pdfs.data() + i * sample_map_size.x);
            std::copy_n(
                alias_table.data(), sample_map_size.x,
                aliases.data() + sample_map_size.y + i * sample_map_size.x);
        }
        // construct marginal alias table
        auto [alias_table, pdf_table] = create_alias_table(row_averages);
        std::copy_n(alias_table.data(), sample_map_size.y, aliases.data());
        for (auto y = 0u; y < sample_map_size.y; y++) {
            auto offset = y * sample_map_size.x;
            auto pdf_y = pdf_table[y];
            auto scale = static_cast<float>(pdf_y * pixel_count);
            for (auto x = 0u; x < sample_map_size.x; x++) {
                pdfs[offset + x] *= scale;
            }
        }
        auto [alias_buffer_view, alias_buffer_id] =
            pipeline.bindless_arena_buffer<AliasEntry>(aliases.size());
        auto [pdf_buffer_view, pdf_buffer_id] =
            pipeline.bindless_arena_buffer<float>(pdfs.size());
        command_buffer << alias_buffer_view.copy_from(aliases.data())
                       << pdf_buffer_view.copy_from(pdfs.data())
                       << commit();
        alias_id.emplace(alias_buffer_id);
        pdf_id.emplace(pdf_buffer_id);
    }
    return luisa::make_unique<SphericalInstance>(
        pipeline, this, texture, std::move(alias_id), std::move(pdf_id));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Spherical)
