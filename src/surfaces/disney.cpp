//
// Created by Mike Smith on 2022/1/30.
//

#include <util/sampling.h>
#include <base/surface.h>
#include <base/texture.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class DisneySurface final : public Surface {

private:
    const Texture *_base_color{};
    const Texture *_metallic{};
    const Texture *_roughness{};
    const Texture *_anisotropy{};
    const Texture *_specular{};
    const Texture *_specular_tint{};
    const Texture *_sheen{};
    const Texture *_sheen_tint{};
    const Texture *_clearcoat{};
    const Texture *_clearcoat_gloss{};
    const Texture *_ior{};
    const Texture *_specular_trans{};

public:
    DisneySurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc} {
        auto load_texture = [scene, desc](const Texture *&t, std::string_view name, Texture::Category category) noexcept {
            if (category == Texture::Category::COLOR) {
                t = scene->load_texture(desc->property_node_or_default(
                    name, SceneNodeDesc::shared_default_texture("ConstColor")));
                if (t->category() != category) [[unlikely]] {
                    LUISA_ERROR(
                        "Expected color texture for "
                        "property '{}' in DisneySurface. [{}]",
                        name, desc->source_location().string());
                }
            } else {
                t = scene->load_texture(desc->property_node_or_default(
                    name, SceneNodeDesc::shared_default_texture("ConstGeneric")));
                if (t->category() != category) [[unlikely]] {
                    LUISA_ERROR(
                        "Expected generic texture for "
                        "property '{}' in DisneySurface. [{}]",
                        name, desc->source_location().string());
                }
            }
        };
#define LUISA_RENDER_DISNEY_PARAM_LOAD(name, category) \
    load_texture(_##name, #name, Texture::Category::category);
        LUISA_RENDER_DISNEY_PARAM_LOAD(base_color, COLOR)
        LUISA_RENDER_DISNEY_PARAM_LOAD(metallic, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(roughness, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(anisotropy, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(specular, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(specular_tint, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(sheen, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(sheen_tint, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(clearcoat, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(clearcoat_gloss, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(ior, GENERIC)
        LUISA_RENDER_DISNEY_PARAM_LOAD(specular_trans, GENERIC)
#undef LUISA_RENDER_DISNEY_PARAM_LOAD
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
    [[nodiscard]] uint encode(
        Pipeline &pipeline, CommandBuffer &command_buffer,
        uint instance_id, const Shape *shape) const noexcept override {
        std::array textures{
            *pipeline.encode_texture(command_buffer, _base_color),
            *pipeline.encode_texture(command_buffer, _metallic),
            *pipeline.encode_texture(command_buffer, _roughness),
            *pipeline.encode_texture(command_buffer, _anisotropy),
            *pipeline.encode_texture(command_buffer, _specular),
            *pipeline.encode_texture(command_buffer, _specular_tint),
            *pipeline.encode_texture(command_buffer, _sheen),
            *pipeline.encode_texture(command_buffer, _sheen_tint),
            *pipeline.encode_texture(command_buffer, _clearcoat),
            *pipeline.encode_texture(command_buffer, _clearcoat_gloss),
            *pipeline.encode_texture(command_buffer, _ior),
            *pipeline.encode_texture(command_buffer, _specular_trans)};
        auto [buffer, buffer_id] = pipeline.arena_buffer<TextureHandle>(textures.size());
        command_buffer << buffer.copy_from(&textures) << compute::commit();
        return buffer_id;
    }
    [[nodiscard]] luisa::unique_ptr<Closure> decode(
        const Pipeline &pipeline, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

class DisneySurfaceClosure final : public Surface::Closure {

private:
    Float3 _n;
    Float3 _vx;
    Float3 _vy;
    Float3 _wo;
    SampledWavelengths _swl;
    Float4 _base_color;
    Float _metallic;
    Float _roughness;
    Float _anisotropy;
    Float _specular;
    Float _specular_tint;
    Float _sheen;
    Float _sheen_tint;
    Float _clearcoat;
    Float _clearcoat_gloss;
    Float _ior;
    Float _specular_trans;

public:
    DisneySurfaceClosure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float4> base_color, Expr<float> metallic, Expr<float> roughness, Expr<float> anisotropy,
        Expr<float> specular, Expr<float> specular_tint, Expr<float> sheen, Expr<float> sheen_tint,
        Expr<float> clearcoat, Expr<float> clearcoat_gloss, Expr<float> ior, Expr<float> specular_trans) noexcept
        : _n{it.shading().n()}, _vx{it.shading().u()}, _vy{it.shading().v()}, _wo{it.wo()}, _swl{swl},
          _base_color{base_color}, _metallic{metallic}, _roughness{roughness}, _anisotropy{anisotropy},
          _specular{specular}, _specular_tint{specular_tint}, _sheen{sheen}, _sheen_tint{sheen_tint},
          _clearcoat{clearcoat}, _clearcoat_gloss{clearcoat_gloss},
          _ior{ite(ior == 0.0f, 1.5f, ior)}, _specular_trans{specular_trans} {}

#define LUISA_RENDER_DISNEY_PARAM_GETTER(name) \
    [[nodiscard]] const auto &name() const noexcept { return _##name; }
    LUISA_RENDER_DISNEY_PARAM_GETTER(base_color)
    LUISA_RENDER_DISNEY_PARAM_GETTER(metallic)
    LUISA_RENDER_DISNEY_PARAM_GETTER(roughness)
    LUISA_RENDER_DISNEY_PARAM_GETTER(anisotropy)
    LUISA_RENDER_DISNEY_PARAM_GETTER(specular)
    LUISA_RENDER_DISNEY_PARAM_GETTER(specular_tint)
    LUISA_RENDER_DISNEY_PARAM_GETTER(sheen)
    LUISA_RENDER_DISNEY_PARAM_GETTER(sheen_tint)
    LUISA_RENDER_DISNEY_PARAM_GETTER(clearcoat)
    LUISA_RENDER_DISNEY_PARAM_GETTER(clearcoat_gloss)
    LUISA_RENDER_DISNEY_PARAM_GETTER(ior)
    LUISA_RENDER_DISNEY_PARAM_GETTER(specular_trans)
#undef LUISA_RENDER_DISNEY_PARAM_GETTER
    [[nodiscard]] const auto &n() const noexcept { return _n; }
    [[nodiscard]] const auto &vx() const noexcept { return _vx; }
    [[nodiscard]] const auto &vy() const noexcept { return _vy; }
    [[nodiscard]] const auto &wo() const noexcept { return _wo; }
    [[nodiscard]] const auto &swl() const noexcept { return _swl; }
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override;
    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override;
};

luisa::unique_ptr<Surface::Closure> DisneySurface::decode(
    const Pipeline &pipeline, const Interaction &it,
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto params = pipeline.buffer<TextureHandle>(it.shape()->surface_buffer_id());
    return luisa::make_unique<DisneySurfaceClosure>(
        it, swl,
        pipeline.evaluate_color_texture(params.read(0), it, swl, time),
        pipeline.evaluate_generic_texture(params.read(1), it, time).x /* metallic */,
        pipeline.evaluate_generic_texture(params.read(2), it, time).x /* roughness */,
        pipeline.evaluate_generic_texture(params.read(3), it, time).x /* anisotropy */,
        pipeline.evaluate_generic_texture(params.read(4), it, time).x /* specular */,
        pipeline.evaluate_generic_texture(params.read(5), it, time).x /* specular_tint */,
        pipeline.evaluate_generic_texture(params.read(6), it, time).x /* sheen */,
        pipeline.evaluate_generic_texture(params.read(7), it, time).x /* sheen_tint */,
        pipeline.evaluate_generic_texture(params.read(8), it, time).x /* clearcoat */,
        pipeline.evaluate_generic_texture(params.read(9), it, time).x /* clearcoat_gloss */,
        pipeline.evaluate_generic_texture(params.read(10), it, time).x /* ior */,
        pipeline.evaluate_generic_texture(params.read(11), it, time).x /* specular_trans */);
}

// from: https://github.com/Twinklebear/ChameleonRT/blob/master/backends/optix/disney_bsdf.h
namespace detail {

using namespace luisa::compute;

[[nodiscard]] inline auto same_hemisphere(const Float3 &w_o, const Float3 &w_i, const Float3 &n) noexcept {
    return dot(w_o, n) * dot(w_i, n) > 0.f;
}

[[nodiscard]] inline auto spherical_dir(Float sin_theta, Float cos_theta, Float phi) noexcept {
    return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

[[nodiscard]] inline auto pow2(auto x) noexcept {
    return x * x;
}

[[nodiscard]] inline auto schlick_weight(Float cos_theta) noexcept {
    auto c = saturate(1.0f - cos_theta);
    auto cc = c * c;
    return cc * cc * c;
}

[[nodiscard]] inline auto fresnel_dielectric(Float cos_theta_i, Float eta_i, Float eta_t) noexcept {
    auto g = pow2(eta_t) / pow2(eta_i) - 1.f + pow2(cos_theta_i);
    auto f = def(1.f);
    $if(g >= 0.0f) {
        f = 0.5f * pow2(g - cos_theta_i) / pow2(g + cos_theta_i) *
            (1.f + pow2(cos_theta_i * (g + cos_theta_i) - 1.f) /
                       pow2(cos_theta_i * (g - cos_theta_i) + 1.f));
    };
    return f;
}

[[nodiscard]] inline auto gtr_1(Float cos_theta_h, Float alpha) noexcept {
    auto g = def(inv_pi);
    $if(alpha < 1.0f) {
        auto alpha_sqr = alpha * alpha;
        g = inv_pi * (alpha_sqr - 1.f) /
            (log(alpha_sqr) * (1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h));
    };
    return g;
}

[[nodiscard]] inline auto gtr_2(Float cos_theta_h, Float alpha) noexcept {
    auto alpha_sqr = alpha * alpha;
    return inv_pi * alpha_sqr /
           pow2(1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h);
}

[[nodiscard]] inline auto gtr_2_aniso(Float h_dot_n, Float h_dot_x, Float h_dot_y, Float2 alpha) noexcept {
    auto denom = pi * alpha.x * alpha.y *
                 pow2(pow2(h_dot_x / alpha.x) + pow2(h_dot_y / alpha.y) + h_dot_n * h_dot_n);
    return 1.0f / denom;
}

[[nodiscard]] inline auto smith_shadowing_ggx(Float n_dot_o, Float alpha_g) noexcept {
    auto a = alpha_g * alpha_g;
    auto b = n_dot_o * n_dot_o;
    return 1.f / (n_dot_o + sqrt(a + b - a * b));
}

[[nodiscard]] inline auto smith_shadowing_ggx_aniso(Float n_dot_o, Float o_dot_x, Float o_dot_y, Float2 alpha) noexcept {
    return 1.f / (n_dot_o + sqrt(pow2(o_dot_x * alpha.x) + pow2(o_dot_y * alpha.y) + pow2(n_dot_o)));
}

[[nodiscard]] inline auto sample_lambertian_dir(const Float3 &n, const Float3 &v_x, const Float3 &v_y, const Float2 &s) noexcept {
    auto hemi_dir = normalize(sample_cosine_hemisphere(s));
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

[[nodiscard]] inline auto sample_gtr_1_h(const Float3 &n, const Float3 &v_x, const Float3 &v_y, Float alpha, const Float2 &s) noexcept {
    auto phi_h = 2.f * pi * s.x;
    auto alpha_sqr = alpha * alpha;
    auto cos_theta_h_sqr = (1.f - pow(alpha_sqr, 1.f - s.y)) / (1.f - alpha_sqr);
    auto cos_theta_h = sqrt(cos_theta_h_sqr);
    auto sin_theta_h = 1.f - cos_theta_h_sqr;
    auto hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

[[nodiscard]] inline auto sample_gtr_2_h(const Float3 &n, const Float3 &v_x, const Float3 &v_y, Float alpha, const Float2 &s) noexcept {
    auto phi_h = 2.f * pi * s.x;
    auto cos_theta_h_sqr = (1.f - s.y) / (1.f + (alpha * alpha - 1.f) * s.y);
    auto cos_theta_h = sqrt(cos_theta_h_sqr);
    auto sin_theta_h = 1.f - cos_theta_h_sqr;
    auto hemi_dir = normalize(spherical_dir(sin_theta_h, cos_theta_h, phi_h));
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

[[nodiscard]] inline auto sample_gtr_2_aniso_h(const Float3 &n, const Float3 &v_x, const Float3 &v_y, const Float2 &alpha, const Float2 &s) noexcept {
    auto x = 2.f * pi * s.x;
    auto w_h = sqrt(s.y / (1.f - s.y)) * (alpha.x * cos(x) * v_x + alpha.y * sin(x) * v_y) + n;
    return normalize(w_h);
}

[[nodiscard]] inline auto lambertian_pdf(const Float3 &w_o, const Float3 &w_i, const Float3 &n) noexcept {
    auto pdf = abs(inv_pi * dot(w_i, n));
    return ite(same_hemisphere(w_o, w_i, n), pdf, 0.0f);
}

[[nodiscard]] inline auto gtr_1_pdf(const Float3 &w_o, const Float3 &w_i, const Float3 &n, Float alpha) noexcept {
    auto pdf = def(0.0f);
    $if(same_hemisphere(w_o, w_i, n)) {
        auto w_h = normalize(w_i + w_o);
        auto cos_theta_h = dot(n, w_h);
        auto d = gtr_1(cos_theta_h, alpha);
        pdf = d * cos_theta_h / (4.f * dot(w_o, w_h));
    };
    return pdf;
}

[[nodiscard]] inline auto gtr_2_pdf(const Float3 &w_o, const Float3 &w_i, const Float3 &n, Float alpha) noexcept {
    auto pdf = def(0.0f);
    $if(same_hemisphere(w_o, w_i, n)) {
        auto w_h = normalize(w_i + w_o);
        auto cos_theta_h = dot(n, w_h);
        auto d = gtr_2(cos_theta_h, alpha);
        pdf = d * cos_theta_h / (4.f * dot(w_o, w_h));
    };
    return pdf;
}

[[nodiscard]] inline auto gtr_2_transmission_pdf(
    const Float3 &w_o, const Float3 &w_i, const Float3 &n,
    Float alpha, Float ior) noexcept {
    auto pdf = def(0.0f);
    $if(!same_hemisphere(w_o, w_i, n)) {
        auto entering = dot(w_o, n) > 0.f;
        auto eta_o = ite(entering, 1.f, ior);
        auto eta_i = ite(entering, ior, 1.f);
        auto w_h = normalize(w_o + w_i * eta_i / eta_o);
        auto cos_theta_h = abs(dot(n, w_h));
        auto i_dot_h = dot(w_i, w_h);
        auto o_dot_h = dot(w_o, w_h);
        auto d = gtr_2(cos_theta_h, alpha);
        auto dwh_dwi = o_dot_h * pow2(eta_o) /
                       pow2(eta_o * o_dot_h + eta_i * i_dot_h);
        pdf = d * cos_theta_h * abs(dwh_dwi);
    };
    return pdf;
}

[[nodiscard]] inline auto gtr_2_aniso_pdf(
    const Float3 &w_o, const Float3 &w_i, const Float3 &n,
    const Float3 &v_x, const Float3 &v_y, const Float2 &alpha) noexcept {
    auto pdf = def(0.0f);
    $if(same_hemisphere(w_o, w_i, n)) {
        auto w_h = normalize(w_i + w_o);
        auto cos_theta_h = dot(n, w_h);
        auto d = gtr_2_aniso(cos_theta_h, abs(dot(w_h, v_x)), abs(dot(w_h, v_y)), alpha);
        pdf = d * cos_theta_h / (4.f * dot(w_o, w_h));
    };
    return pdf;
}

[[nodiscard]] inline auto disney_diffuse(
    const DisneySurfaceClosure &mat, const Float3 &n,
    const Float3 &w_o, const Float3 &w_i) noexcept {
    Float3 w_h = normalize(w_i + w_o);
    Float n_dot_o = abs(dot(w_o, n));
    Float n_dot_i = abs(dot(w_i, n));
    Float i_dot_h = dot(w_i, w_h);
    Float fd90 = 0.5f + 2.f * mat.roughness() * i_dot_h * i_dot_h;
    Float fi = schlick_weight(n_dot_i);
    Float fo = schlick_weight(n_dot_o);
    return mat.base_color() * inv_pi *
           lerp(1.f, fd90, fi) * lerp(1.f, fd90, fo);
}

[[nodiscard]] inline auto disney_microfacet_isotropic(
    const DisneySurfaceClosure &mat, const Float3 &n,
    const Float3 &w_o, const Float3 &w_i) noexcept {
    auto w_h = normalize(w_i + w_o);
    auto lum = mat.swl().cie_y(mat.base_color());
    auto tint = ite(lum > 0.f, mat.base_color() / lum, make_float4(1.f));
    auto spec = lerp(
        mat.specular() * 0.08f * lerp(1.0f, tint, mat.specular_tint()),
        mat.base_color(),
        mat.metallic());
    auto alpha = max(0.001f, mat.roughness() * mat.roughness());
    auto d = gtr_2(dot(n, w_h), alpha);
    auto f = lerp(spec, 1.0f, schlick_weight(dot(w_i, w_h)));
    auto g = smith_shadowing_ggx(dot(n, w_i), alpha) *
             smith_shadowing_ggx(dot(n, w_o), alpha);
    return d * f * g;
}

[[nodiscard]] inline auto disney_microfacet_transmission_isotropic(
    const DisneySurfaceClosure &mat, const Float3 &n,
    const Float3 &w_o, const Float3 &w_i) noexcept {
    auto o_dot_n = dot(w_o, n);
    auto i_dot_n = dot(w_i, n);
    auto f = def(make_float4(0.0f));
    $if(o_dot_n != 0.0f & i_dot_n != 0.0f) {
        auto entering = o_dot_n > 0.f;
        auto eta_o = ite(entering, 1.f, mat.ior());
        auto eta_i = ite(entering, mat.ior(), 1.f);
        auto w_h = normalize(w_o + w_i * eta_i / eta_o);
        auto alpha = max(0.001f, mat.roughness() * mat.roughness());
        auto d = gtr_2(abs(dot(n, w_h)), alpha);
        auto f = fresnel_dielectric(abs(dot(w_i, n)), eta_o, eta_i);
        auto g = smith_shadowing_ggx(abs(dot(n, w_i)), alpha) *
                 smith_shadowing_ggx(abs(dot(n, w_o)), alpha);
        auto i_dot_h = dot(w_i, w_h);
        auto o_dot_h = dot(w_o, w_h);
        auto c = abs(o_dot_h) / abs(o_dot_n) *
                 abs(i_dot_h) / abs(i_dot_n) *
                 pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h);
        f = mat.base_color() * c * (1.f - f) * g * d;
    };
    return f;
}

[[nodiscard]] inline auto disney_microfacet_anisotropic(
    const DisneySurfaceClosure &mat, const Float3 &n,
    const Float3 &w_o, const Float3 &w_i, const Float3 &v_x, const Float3 &v_y) noexcept {
    auto w_h = normalize(w_i + w_o);
    auto lum = mat.swl().cie_y(mat.base_color());
    auto tint = ite(lum > 0.f, mat.base_color() / lum, 1.0f);
    auto spec = lerp(
        mat.specular() * 0.08f * lerp(1.0f, tint, mat.specular_tint()),
        mat.base_color(), mat.metallic());
    auto aspect = sqrt(1.f - mat.anisotropy() * 0.9f);
    auto a = mat.roughness() * mat.roughness();
    auto alpha = make_float2(max(0.001f, a / aspect), max(0.001f, a * aspect));
    auto d = gtr_2_aniso(dot(n, w_h), abs(dot(w_h, v_x)), abs(dot(w_h, v_y)), alpha);
    auto f = lerp(spec, 1.0f, schlick_weight(dot(w_i, w_h)));
    auto g = smith_shadowing_ggx_aniso(dot(n, w_i), abs(dot(w_i, v_x)), abs(dot(w_i, v_y)), alpha) *
             smith_shadowing_ggx_aniso(dot(n, w_o), abs(dot(w_o, v_x)), abs(dot(w_o, v_y)), alpha);
    return d * f * g;
}

[[nodiscard]] inline auto disney_clear_coat(
    const DisneySurfaceClosure &mat, const Float3 &n,
    const Float3 &w_o, const Float3 &w_i) noexcept {
    auto w_h = normalize(w_i + w_o);
    auto alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss());
    auto d = gtr_1(dot(n, w_h), alpha);
    auto f = lerp(0.04f, 1.f, schlick_weight(dot(w_i, n)));
    auto g = smith_shadowing_ggx(dot(n, w_i), 0.25f) *
             smith_shadowing_ggx(dot(n, w_o), 0.25f);
    return 0.25f * mat.clearcoat() * d * f * g;
}

[[nodiscard]] inline auto disney_sheen(
    const DisneySurfaceClosure &mat, const Float3 &n,
    const Float3 &w_o, const Float3 &w_i) noexcept {
    auto w_h = normalize(w_i + w_o);
    auto lum = mat.swl().cie_y(mat.base_color());
    auto tint = ite(lum > 0.f, mat.base_color() / lum, 1.0f);
    auto sheen_color = lerp(1.0f, tint, mat.sheen_tint());
    auto f = schlick_weight(dot(w_i, n));
    return f * mat.sheen() * sheen_color;
}

[[nodiscard]] auto disney_surface_f(const DisneySurfaceClosure &mat, const Float3 &w_i) noexcept {
    auto f = def(make_float4());
    $if(same_hemisphere(mat.wo(), w_i, mat.n())) {
        auto coat = disney_clear_coat(mat, mat.n(), mat.wo(), w_i);
        auto sheen = disney_sheen(mat, mat.n(), mat.wo(), w_i);
        auto diffuse = disney_diffuse(mat, mat.n(), mat.wo(), w_i);
        auto gloss = def(make_float4());
        $if(mat.anisotropy() == 0.f) {
            gloss = disney_microfacet_isotropic(
                mat, mat.n(), mat.wo(), w_i);
        }
        $else {
            gloss = disney_microfacet_anisotropic(
                mat, mat.n(), mat.wo(), w_i, mat.vx(), mat.vy());
        };
        f = (diffuse + sheen) * (1.f - mat.metallic()) * (1.f - mat.specular_trans()) + gloss + coat;
    }
    $elif(mat.specular_trans() > 0.f) {
        auto spec_trans = disney_microfacet_transmission_isotropic(
            mat, mat.n(), mat.wo(), w_i);
        f = spec_trans * (1.f - mat.metallic()) * mat.specular_trans();
    };
    return f;
}

[[nodiscard]] auto disney_surface_pdf(const DisneySurfaceClosure &mat, const Float3 &w_i) noexcept {
    auto alpha = max(0.001f, mat.roughness() * mat.roughness());
    auto aspect = sqrt(1.f - mat.anisotropy() * 0.9f);
    auto alpha_aniso = make_float2(
        max(0.001f, alpha / aspect),
        max(0.001f, alpha * aspect));
    auto clearcoat_alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss());
    auto pdf = lambertian_pdf(mat.wo(), w_i, mat.n()) +
               gtr_1_pdf(mat.wo(), w_i, mat.n(), clearcoat_alpha);
    $if(mat.anisotropy() == 0.f) {
        pdf += gtr_2_pdf(mat.wo(), w_i, mat.n(), alpha);
    }
    $else {
        pdf += gtr_2_aniso_pdf(
            mat.wo(), w_i, mat.n(),
            mat.vx(), mat.vy(), alpha_aniso);
    };
    auto inv_n = def(1.0f / 3.0f);
    $if(mat.specular_trans() > 0.f) {
        inv_n = 1.0f / 4.0f;
        pdf += gtr_2_transmission_pdf(
            mat.wo(), w_i, mat.n(), alpha, mat.ior());
    };
    return pdf * inv_n;
}

[[nodiscard]] inline auto reflect(const Float3 &i, const Float3 &n) noexcept {
    return i - 2.0f * dot(n, i) * n;
}

[[nodiscard]] inline auto refract(const Float3 &i, const Float3 &n, Float eta) noexcept {
    auto c = dot(n, i);
    auto k = 1.0f - eta * eta * (1.0f - c * c);
    return ite(k < 0.0f, 0.0f, eta * i - (eta * c + sqrt(k)) * n);
}

[[nodiscard]] std::tuple<Float3 /* wi */, Float4 /* f */, Float /* pdf */>
disney_surface_sample(const DisneySurfaceClosure &mat, const Float2 &u_in) noexcept {
    auto num_comp = ite(mat.specular_trans() == 0.f, 3.0f, 4.0f);
    auto component = cast<uint>(min(u_in.x * num_comp, num_comp - 1.0f));
    auto u = make_float2(u_in.x * num_comp - cast<float>(component), u_in.y);
    auto n = mat.n();
    auto wo = mat.wo();
    auto vx = mat.vx();
    auto vy = mat.vy();
    auto wi = def(make_float3());
    $switch(component) {
        $case(0u) {
            wi = sample_lambertian_dir(n, vx, vy, u);
        };
        $case(1u) {
            auto wh = def(make_float3());
            auto alpha = max(0.001f, mat.roughness() * mat.roughness());
            $if(mat.anisotropy() == 0.f) {
                wh = sample_gtr_2_h(n, vx, vy, alpha, u);
            }
            $else {
                auto aspect = sqrt(1.f - mat.anisotropy() * 0.9f);
                auto alpha_aniso = make_float2(
                    max(0.001f, alpha / aspect),
                    max(0.001f, alpha * aspect));
                wh = sample_gtr_2_aniso_h(n, vx, vy, alpha_aniso, u);
            };
            wi = reflect(-wo, wh);
            wi = ite(same_hemisphere(wo, wi, n), wi, make_float3(0.0f));
        };
        $case(2u) {
            auto alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss());
            auto wh = sample_gtr_1_h(n, vx, vy, alpha, u);
            wi = reflect(-wo, wh);
            wi = ite(same_hemisphere(wo, wi, n), wi, make_float3(0.0f));
        };
        $case(3u) {
            auto alpha = max(0.001f, mat.roughness() * mat.roughness());
            auto wh = sample_gtr_2_h(n, vx, vy, alpha, u);
            wh = sign(dot(wo, wh)) * wh;
            auto entering = dot(wo, n) > 0.f;
            wi = refract(-wo, wh, ite(entering, 1.f / mat.ior(), mat.ior()));
        };
        $default { unreachable(); };
    };
    auto pdf = def(0.0f);
    auto f = def(make_float4());
    $if(!all(wi == 0.0f)) {
        f = disney_surface_f(mat, wi);
        pdf = disney_surface_pdf(mat, wi);
    };
    return std::make_tuple(wi, f, pdf);
}

}// namespace detail

Surface::Evaluation DisneySurfaceClosure::evaluate(Expr<float3> wi) const noexcept {
    return {.f = detail::disney_surface_f(*this, wi),
            .pdf = detail::disney_surface_pdf(*this, wi)};
}

Surface::Sample DisneySurfaceClosure::sample(Sampler::Instance &sampler) const noexcept {
    auto u = sampler.generate_2d();
    auto [wi, f, pdf] = detail::disney_surface_sample(*this, u);
    return {.wi = wi, .eval = {.f = f, .pdf = pdf}};
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::DisneySurface)
