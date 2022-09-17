//
// Created by Mike Smith on 2022/1/9.
//

#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

#include <utility>

namespace luisa::render {

namespace ior {

static constexpr auto lut_step = 5u;
static constexpr auto lut_min = static_cast<uint>(visible_wavelength_min);
static constexpr auto lut_max = static_cast<uint>(visible_wavelength_max);
static constexpr auto lut_size = (lut_max - lut_min) / lut_step + 1;

#include <surfaces/metal_ior.inl.h>

}// namespace ior

using namespace luisa::compute;

class MetalSurface final : public Surface {

public:
    struct ComplexIOR {
        luisa::vector<float> eta;
        luisa::vector<float> k;
    };

private:
    const Texture *_roughness;
    const Texture *_kd;
    luisa::string _ior;
    bool _remap_roughness;

public:
    [[nodiscard]] static auto &_mutex() noexcept {
        static std::mutex mutex;
        return mutex;
    }
    [[nodiscard]] static auto &_known_ior() noexcept {
        static luisa::unordered_map<luisa::string, ComplexIOR> known_ior;
        return known_ior;
    }

public:
    MetalSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _kd{scene->load_texture(desc->property_node_or_default("Kd"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        auto register_eta_k = [&](luisa::string_view name, luisa::span<float2> eta_k) noexcept {
            std::scoped_lock lock{_mutex()};
            auto [iter, success] = _known_ior().try_emplace(name, ComplexIOR{});
            if (success) {
                auto &ior = iter->second;
                ior.eta.resize(eta_k.size());
                ior.k.resize(eta_k.size());
                for (auto i = 0u; i < eta_k.size(); i++) {
                    ior.eta[i] = eta_k[i].x;
                    ior.k[i] = eta_k[i].y;
                }
            }
            return iter->first;
        };
        if (auto eta_name = desc->property_string_or_default("eta"); !eta_name.empty()) {
            for (auto &c : eta_name) { c = static_cast<char>(tolower(c)); }
            if (eta_name == "ag" || eta_name == "silver") {
                _ior = register_eta_k("__internal_ior_Ag", ior::Ag);
            } else if (eta_name == "al" || eta_name == "aluminium") {
                _ior = register_eta_k("__internal_ior_Al", ior::Al);
            } else if (eta_name == "au" || eta_name == "gold") {
                _ior = register_eta_k("__internal_ior_Au", ior::Au);
            } else if (eta_name == "cu" || eta_name == "copper") {
                _ior = register_eta_k("__internal_ior_Cu", ior::Cu);
            } else if (eta_name == "cuzn" || eta_name == "cu-zn" || eta_name == "brass") {
                _ior = register_eta_k("__internal_ior_CuZn", ior::CuZn);
            } else if (eta_name == "fe" || eta_name == "iron") {
                _ior = register_eta_k("__internal_ior_Fe", ior::Fe);
            } else if (eta_name == "ti" || eta_name == "titanium") {
                _ior = register_eta_k("__internal_ior_Ti", ior::Ti);
            } else if (eta_name == "v" || eta_name == "vanadium") {
                _ior = register_eta_k("__internal_ior_V", ior::V);
            } else if (eta_name == "vn") {
                _ior = register_eta_k("__internal_ior_VN", ior::VN);
            } else if (eta_name == "li" || eta_name == "lithium") {
                _ior = register_eta_k("__internal_ior_Li", ior::Li);
            } else [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Unknown metal '{}'. "
                    "Fallback to Aluminium. [{}]",
                    eta_name,
                    desc->source_location().string());
                _ior = register_eta_k("__internal_ior_Al", ior::Al);
            }
        } else {
            auto eta = desc->property_float_list("eta");
            if (eta.size() % 3u != 0u) [[unlikely]] {
                LUISA_ERROR(
                    "Invalid eta list size: {}. [{}]",
                    eta.size(), desc->source_location().string());
            }
            luisa::vector<float> lambda(eta.size() / 3u);
            luisa::vector<float> n(eta.size() / 3u);
            luisa::vector<float> k(eta.size() / 3u);
            for (auto i = 0u; i < eta.size() / 3u; i++) {
                lambda[i] = eta[i * 3u + 0u];
                n[i] = eta[i * 3u + 1u];
                k[i] = eta[i * 3u + 2u];
            }
            if (!std::is_sorted(lambda.cbegin(), lambda.cend())) [[unlikely]] {
                LUISA_ERROR(
                    "Unsorted wavelengths in eta list. [{}]",
                    desc->source_location().string());
            }
            if (lambda.front() > visible_wavelength_min ||
                lambda.back() < visible_wavelength_max) [[unlikely]] {
                LUISA_ERROR(
                    "Invalid wavelength range [{}, {}] in eta list. [{}]",
                    lambda.front(), lambda.back(), desc->source_location().string());
            }
            // TODO: scan rather than binary search
            luisa::vector<float2> lut(ior::lut_size);
            for (auto i = 0u; i < ior::lut_size; i++) {
                auto wavelength = static_cast<float>(
                    i * ior::lut_step + ior::lut_min);
                auto lb = std::lower_bound(
                    lambda.cbegin(), lambda.cend(), wavelength);
                auto index = std::clamp(
                    static_cast<size_t>(std::distance(lambda.cbegin(), lb)),
                    static_cast<size_t>(1u), lambda.size() - 1u);
                auto t = (wavelength - lambda[index - 1u]) /
                         (lambda[index] - lambda[index - 1u]);
                lut[i] = make_float2(
                    std::lerp(n[index - 1u], n[index], t),
                    std::lerp(k[index - 1u], k[index], t));
            }
            auto name = luisa::format(
                "__custom_ior_{:016x}",
                luisa::detail::murmur2_hash64(
                    eta.data(), eta.size() * sizeof(float), 19980810u));
            _ior = register_eta_k(name, lut);
        }
        LUISA_RENDER_CHECK_GENERIC_TEXTURE(MetalSurface, roughness, 1);
        LUISA_RENDER_CHECK_ALBEDO_TEXTURE(MetalSurface, kd);
    }
    [[nodiscard]] auto ior() const noexcept { return _ior; }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MetalInstance final : public Surface::Instance {

private:
    const Texture::Instance *_roughness;
    const Texture::Instance *_kd;
    SPD _eta;
    SPD _k;

public:
    MetalInstance(const Pipeline &pipeline, const Surface *surface,
                  const Texture::Instance *roughness, const Texture::Instance *Kd,
                  SPD eta, SPD k) noexcept
        : Surface::Instance{pipeline, surface},
          _roughness{roughness}, _kd{Kd}, _eta{eta}, _k{k} {}

private:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl,
        Expr<float> eta_i, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MetalSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    auto &&eta_k = _known_ior().at(_ior);
    auto eta_buffer_id = pipeline.register_named_id(luisa::format("{}.eta", _ior), [&] {
        auto [eta_view, eta_id] = pipeline.arena_buffer<float>(eta_k.eta.size());
        command_buffer << eta_view.copy_from(eta_k.eta.data());
        return eta_id;
    });
    auto k_buffer_id = pipeline.register_named_id(luisa::format("{}.k", _ior), [&] {
        auto [k_view, k_id] = pipeline.arena_buffer<float>(eta_k.k.size());
        command_buffer << k_view.copy_from(eta_k.k.data());
        return k_id;
    });
    return luisa::make_unique<MetalInstance>(
        pipeline, this, roughness, Kd,
        SPD{pipeline, eta_buffer_id, static_cast<float>(ior::lut_step)},
        SPD{pipeline, k_buffer_id, static_cast<float>(ior::lut_step)});
}

class MetalClosure final : public Surface::Closure {

private:
    luisa::optional<SampledSpectrum> _refl;
    luisa::unique_ptr<FresnelConductor> _fresnel;
    luisa::unique_ptr<TrowbridgeReitzDistribution> _distrib;
    luisa::unique_ptr<MicrofacetReflection> _lobe;

public:
    MetalClosure(
        const Surface::Instance *instance, const Interaction &it,
        const SampledWavelengths &swl, Expr<float> time,
        Expr<float> eta_i, const SampledSpectrum &n, const SampledSpectrum &k,
        luisa::optional<SampledSpectrum> refl, Expr<float2> alpha) noexcept
        : Surface::Closure{instance, it, swl, time}, _refl{std::move(refl)},
          _fresnel{luisa::make_unique<FresnelConductor>(eta_i, n, k)},
          _distrib{luisa::make_unique<TrowbridgeReitzDistribution>(alpha)},
          _lobe{luisa::make_unique<MicrofacetReflection>(SampledSpectrum{swl.dimension(), 1.f}, _distrib.get(), _fresnel.get())} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = _lobe->evaluate(wo_local, wi_local);
        if (_refl) { f *= *_refl; }
        auto pdf = _lobe->pdf(wo_local, wi_local);
        return {.f = f * abs_cos_theta(wi_local),
                .pdf = pdf,
                .roughness = _distrib->alpha(),
                .eta = _fresnel->eta_i()};
    }
    [[nodiscard]] Surface::Sample sample(Expr<float>, Expr<float2> u) const noexcept override {
        auto wo_local = _it.wo_local();
        auto pdf = def(0.f);
        auto wi_local = def(make_float3(0.f, 0.f, 1.f));
        auto f = _lobe->sample(wo_local, &wi_local, u, &pdf);
        if (_refl) { f *= *_refl; }
        auto wi = _it.shading().local_to_world(wi_local);
        return {.wi = wi,
                .eval = {.f = f * abs_cos_theta(wi_local),
                         .pdf = pdf,
                         .roughness = _distrib->alpha(),
                         .eta = _fresnel->eta_i()}};
    }
    void backward(Expr<float3> wi, const SampledSpectrum &df) const noexcept override {
        // Metal surface is not differentiable
    }
};

luisa::unique_ptr<Surface::Closure> MetalInstance::closure(
    const Interaction &it, const SampledWavelengths &swl,
    Expr<float> eta_i, Expr<float> time) const noexcept {
    auto alpha = def(make_float2(.5f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, swl, time);
        auto remap = node<MetalSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    SampledSpectrum eta{swl.dimension()};
    SampledSpectrum k{swl.dimension()};
    for (auto i = 0u; i < swl.dimension(); i++) {
        auto lambda = swl.lambda(i);
        eta[i] = _eta.sample(lambda);
        k[i] = _k.sample(lambda);
    }
    luisa::optional<SampledSpectrum> refl;
    if (_kd != nullptr) {
        refl.emplace(_kd->evaluate_albedo_spectrum(it, swl, time).value);
    }
    return luisa::make_unique<MetalClosure>(
        this, it, swl, time, eta_i,
        eta, k, std::move(refl), alpha);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MetalSurface)
