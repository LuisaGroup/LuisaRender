//
// Created by Mike Smith on 2022/1/15.
//

#include <numbers>
#include <tinyexr.h>

#include <util/sampling.h>
#include <util/imageio.h>
#include <base/interaction.h>
#include <base/environment.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

class EnvironmentMapping final : public Environment {

public:
    static constexpr auto sample_map_size = make_uint2(256u, 128u);

private:
    const Texture *_emission;
    float _scale;

public:
    EnvironmentMapping(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Environment{scene, desc},
          _emission{scene->load_texture(desc->property_node_or_default(
              "emission", SceneNodeDesc::shared_default_texture("ConstIllum")))},
          _scale{std::max(desc->property_float_or_default("scale", 1.0f), 0.0f)} {
        if (_emission->category() != Texture::Category::ILLUMINANT) [[unlikely]] {
            LUISA_ERROR(
                "Non-illuminant textures are not "
                "allowed in environment mapping. [{}]",
                desc->source_location().string());
        }
    }
    [[nodiscard]] auto scale() const noexcept { return _scale; }
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
        return normalize(make_float3(x, y, z));
    }
    [[nodiscard]] static auto direction_to_uv(Expr<float3> w) noexcept {
        auto theta = acos(w.y);
        auto phi = atan2(w.x, w.z);
        auto u = 1.f - 0.5f * inv_pi * phi;
        auto v = theta * inv_pi;
        return fract(make_float2(u, v));
    }
};

class EnvironmentMappingInstance final : public Environment::Instance {

private:
    const Texture::Instance *_texture;
    luisa::optional<uint> _alias_buffer_id;
    luisa::optional<uint> _pdf_buffer_id;

private:
    [[nodiscard]] auto _evaluate(Expr<float3> wi_local, const SampledWavelengths &swl, Expr<float> time) const noexcept {
        auto env = node<EnvironmentMapping>();
        auto uv = EnvironmentMapping::direction_to_uv(wi_local);
        Interaction it{-wi_local, uv};
        auto L = _texture->evaluate(it, swl, time).value;
        return L * env->scale();
    }

public:
    EnvironmentMappingInstance(
        const Pipeline &pipeline, const Environment *env, const Texture::Instance *texture,
        luisa::optional<uint> alias_buffer_id, luisa::optional<uint> pdf_buffer_id) noexcept
        : Environment::Instance{pipeline, env}, _texture{texture},
          _alias_buffer_id{std::move(alias_buffer_id)},
          _pdf_buffer_id{std::move(pdf_buffer_id)} {}
    [[nodiscard]] Light::Evaluation evaluate(
        Expr<float3> wi, Expr<float3x3> env_to_world,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto world_to_env = transpose(env_to_world);
        auto wi_local = world_to_env * wi;
        auto L = _evaluate(wi_local, swl, time);
        if (_texture->node()->is_constant()) {
            return {.L = L, .pdf = uniform_sphere_pdf()};
        }
        auto uv = EnvironmentMapping::direction_to_uv(wi_local);
        auto size = make_float2(EnvironmentMapping::sample_map_size);
        auto ix = cast<uint>(clamp(uv.x * size.x, 0.f, size.x - 1.f));
        auto iy = cast<uint>(clamp(uv.y * size.y, 0.f, size.y - 1.f));
        auto pdf_buffer = pipeline().bindless_buffer<float>(*_pdf_buffer_id);
        auto pdf = pdf_buffer.read(iy * EnvironmentMapping::sample_map_size.x + ix);
        return {.L = L, .pdf = pdf};
    }
    [[nodiscard]] Light::Sample sample(
        Sampler::Instance &sampler, Expr<float3> p_from, Expr<float3x3> env_to_world,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override {
        auto [wi_local, pdf] = [&] {
            auto u = sampler.generate_2d();
            if (_texture->node()->is_constant()) {
                return std::make_pair(sample_uniform_sphere(u), def(uniform_sphere_pdf()));
            }
            auto alias_buffer = pipeline().bindless_buffer<AliasEntry>(*_alias_buffer_id);
            auto [iy, uy] = sample_alias_table(
                alias_buffer, EnvironmentMapping::sample_map_size.y, u.y);
            auto offset = EnvironmentMapping::sample_map_size.y +
                          iy * EnvironmentMapping::sample_map_size.x;
            auto [ix, ux] = sample_alias_table(
                alias_buffer, EnvironmentMapping::sample_map_size.x, u.x, offset);
            auto uv = make_float2(cast<float>(ix) + ux, cast<float>(iy) + uy) /
                      make_float2(EnvironmentMapping::sample_map_size);
            auto p = pipeline().bindless_buffer<float>(*_pdf_buffer_id).read(iy * EnvironmentMapping::sample_map_size.x + ix);
            return std::make_pair(EnvironmentMapping::uv_to_direction(uv), p);
        }();
        return {.eval = {.L = _evaluate(wi_local, swl, time), .pdf = pdf},
                .wi = normalize(env_to_world * wi_local),
                .distance = std::numeric_limits<float>::max()};
    }
};

unique_ptr<Environment::Instance> EnvironmentMapping::build(
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
            auto coord = dispatch_id().xy();
            auto uv = (make_float2(coord) + .5f) /
                      make_float2(sample_map_size);
            auto w = EnvironmentMapping::uv_to_direction(uv);
            auto it = Interaction{-w, uv};
            auto scale = texture->evaluate(it, {}, 0.f).scale;
            auto sin_theta = sin(uv.y * pi);
            auto pixel_id = coord.y * sample_map_size.x + coord.x;
            scale_map_device.write(pixel_id, max(sin_theta * scale, 1e-2f));
        };
        auto generate_weight_map = device.compile(generate_weight_map_kernel);
        command_buffer << generate_weight_map().dispatch(sample_map_size)
                       << scale_map_device.copy_to(scale_map.data())
                       << synchronize();
        auto sum_scale = 0.;
        for (auto s : scale_map) { sum_scale += s; }
        auto average_scale = static_cast<float>(sum_scale / pixel_count);
        for (auto &&s : scale_map) { s = std::max(s - average_scale, 1e-2f); }
        luisa::vector<float> row_averages(sample_map_size.y);
        luisa::vector<float> pdfs(pixel_count);
        luisa::vector<AliasEntry> aliases(sample_map_size.y + pixel_count);
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
        auto [alias_table, pdf_table] = create_alias_table(row_averages);
        std::copy_n(alias_table.data(), sample_map_size.y, aliases.data());
        for (auto y = 0u; y < sample_map_size.y; y++) {
            auto offset = y * sample_map_size.x;
            auto pdf_y = pdf_table[y];
            auto scale = static_cast<float>(
                .25 * std::numbers::inv_pi * pdf_y * pixel_count);
            for (auto x = 0u; x < sample_map_size.x; x++) {
                pdfs[offset + x] *= scale;
            }
        }
        auto [alias_buffer_view, alias_buffer_id] =
            pipeline.arena_buffer<AliasEntry>(aliases.size());
        auto [pdf_buffer_view, pdf_buffer_id] =
            pipeline.arena_buffer<float>(pdfs.size());
        command_buffer << alias_buffer_view.copy_from(aliases.data())
                       << pdf_buffer_view.copy_from(pdfs.data())
                       << commit();
        alias_id.emplace(alias_buffer_id);
        pdf_id.emplace(pdf_buffer_id);

        auto size = make_int2(sample_map_size);
        SaveEXR(pdfs.data(), size.x, size.y, 1, false, "pdf.exr", nullptr);
    }
    return luisa::make_unique<EnvironmentMappingInstance>(
        pipeline, this, texture, std::move(alias_id), std::move(pdf_id));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::EnvironmentMapping)
