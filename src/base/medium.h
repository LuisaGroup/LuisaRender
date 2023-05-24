//
// Created by ChenXin on 2023/2/13.
//

#pragma once

#include <luisa-compute.h>
#include <util/spec.h>
#include <base/scene_node.h>
#include <base/spectrum.h>
#include <base/phase_function.h>
#include <base/interaction.h>
#include <util/sampling.h>
#include <util/rng.h>

#include <utility>

namespace luisa::render {

using compute::Expr;
using compute::Var;

#define RAY_EPSILON 1e-3f

class Medium : public SceneNode {

public:
    static constexpr uint INVALID_TAG = ~0u;
    static constexpr uint VACUUM_PRIORITY = ~0u;

    static constexpr uint event_absorb = 0u;
    static constexpr uint event_scatter = 1u;
    static constexpr uint event_null = 2u;
    static constexpr uint event_hit_surface = 3u;

    static constexpr uint event_invalid = ~0u;

protected:
    uint _priority;

public:
    [[nodiscard]] static UInt sample_event(Expr<float> p_absorb, Expr<float> p_scatter,
                                           Expr<float> p_null, Expr<float> u) noexcept {
        return sample_discrete(make_float3(p_absorb, p_scatter, p_null), u);
    }

    struct Evaluation {
        SampledSpectrum f;
        Float pdf;

        [[nodiscard]] static auto zero(uint spec_dim) noexcept {
            return Evaluation{
                .f = SampledSpectrum{spec_dim},
                .pdf = 1e16f};
        }
    };

    struct Sample {
        Evaluation eval;
        Var<Ray> ray;
        UInt medium_event;
        Float t;

        [[nodiscard]] static auto zero(uint spec_dim) noexcept {
            return Sample{.eval = Evaluation::zero(spec_dim),
                          .ray = def<Ray>(),
                          .medium_event = event_invalid,
                          .t = 0.0f};
        }
    };

    struct RayMajorantSegment {
        Float t_min, t_max;
        SampledSpectrum sigma_maj;
        Bool empty;

        [[nodiscard]] static auto one(uint spec_dim) noexcept {
            return RayMajorantSegment{
                .t_min = 0.0f,
                .t_max = Interaction::default_t_max,
                .sigma_maj = SampledSpectrum{spec_dim, 1.f},
                .empty = true};
        }
    };

    class RayMajorantIterator {

    public:
        virtual ~RayMajorantIterator() noexcept = default;
        [[nodiscard]] virtual RayMajorantSegment next() noexcept = 0;
    };

    class Instance;

    class Closure {

    private:
        const Instance *_instance;

    private:
        const SampledWavelengths &_swl;
        Var<Ray> _ray;
        Float _time;
        Float _eta;
        SampledSpectrum _sigma_a;// absorption coefficient
        SampledSpectrum _sigma_s;// scattering coefficient
        SampledSpectrum _le;     // emission coefficient
        const PhaseFunction::Instance *_phase_function;

    protected:
        [[nodiscard]] SampledSpectrum analytic_transmittance(
            Expr<float> t,
            const SampledSpectrum &sigma) const noexcept {
            return exp(-sigma * t);
        }

    public:
        Closure(const Instance *instance, Expr<Ray> ray,
                const SampledWavelengths &swl, Expr<float> time, Expr<float> eta,
                const SampledSpectrum &sigma_a, const SampledSpectrum &sigma_s, const SampledSpectrum &le,
                const PhaseFunction::Instance *phase_function) noexcept;
        virtual ~Closure() noexcept = default;
        template<typename T = Instance>
            requires std::is_base_of_v<Instance, T>
        [[nodiscard]] auto instance() const noexcept { return static_cast<const T *>(_instance); }
        [[nodiscard]] auto &swl() const noexcept { return _swl; }
        [[nodiscard]] Var<Ray> ray() const noexcept { return _ray; }
        [[nodiscard]] auto time() const noexcept { return _time; }
        [[nodiscard]] auto eta() const noexcept { return _eta; }
        [[nodiscard]] auto sigma_a() const noexcept { return _sigma_a; }
        [[nodiscard]] auto sigma_s() const noexcept { return _sigma_s; }
        [[nodiscard]] auto sigma_t() const noexcept { return _sigma_a + _sigma_s; }
        [[nodiscard]] auto le() const noexcept { return _le; }
        [[nodiscard]] auto phase_function() const noexcept { return _phase_function; }

        [[nodiscard]] virtual Medium::Sample sample(Expr<float> t_max, PCG32 &rng) const noexcept = 0;
        [[nodiscard]] virtual Medium::Evaluation transmittance(Expr<float> t, PCG32 &rng) const noexcept = 0;
        [[nodiscard]] virtual luisa::unique_ptr<RayMajorantIterator> sample_iterator(Expr<float> t_max) const noexcept = 0;
        // from PBRT-v4
        template<typename F>
        [[nodiscard]] SampledSpectrum sampleT_maj(
            Expr<float> t_max, Expr<float> u, PCG32 &rng,
            //            const luisa::function<Bool(luisa::unique_ptr<Medium::Closure> closure, Expr<float3> p,
            //                                       SampledSpectrum sigma_maj, SampledSpectrum T_maj)> &callback
            F &&callback) const noexcept {
            Float u_local = u;

            // Initialize RayMajorantIterator for ray majorant sampling
            auto majorant_iter = sample_iterator(t_max);

            // Generate ray majorant samples until termination
            SampledSpectrum T_maj(swl().dimension(), 1.f);
            Bool done = def(false);
            $while(!done) {
                // Get next majorant segment from iterator and sample it
                auto seg = majorant_iter->next();
                $if(seg.empty) {
                    done = true;
                    $break;
                };

                // Handle zero-valued majorant for current segment
                $if(seg.sigma_maj[0u] == 0.f) {
                    Float dt = seg.t_max - seg.t_min;
                    // Handle infinite _dt_ for ray majorant segment
                    dt = ite(isinf(dt), std::numeric_limits<float>::max(), dt);

                    T_maj *= exp(-dt * seg.sigma_maj);
                    $continue;
                };

                // Generate samples along current majorant segment
                Float t_min = seg.t_min;
                $while(true) {
                    // Try to generate sample along current majorant segment
                    Float t = t_min + sample_exponential(u_local, seg.sigma_maj[0u]);
                    u_local = rng.uniform_float();
                    $if(t < seg.t_max) {
                        // Call callback function for sample within segment
                        T_maj *= exp(-(t - t_min) * seg.sigma_maj);
                        Float3 p = ray()->origin() + ray()->direction() * t;
                        auto closure_t = instance()->closure(make_ray(p, ray()->direction()), swl(), time());
                        $if(!callback(std::move(closure_t), seg.sigma_maj, T_maj)) {
                            // Returning out of doubly-nested while loop is not as good perf. wise
                            // on the GPU vs using "done" here.
                            done = true;
                            T_maj = SampledSpectrum(1.f);
                            $break;
                        };
                        T_maj = SampledSpectrum(1.f);
                        t_min = t;
                    }
                    $else {
                        // Handle sample past end of majorant segment
                        Float dt = seg.t_max - t_min;
                        // Handle infinite dt for ray majorant segment
                        dt = ite(isinf(dt), std::numeric_limits<float>::max(), dt);

                        T_maj *= exp(-dt * seg.sigma_maj);
                        $break;
                    };
                };
            };

            return T_maj;
        }
    };

    class Instance {
    protected:
        const Pipeline &_pipeline;
        const Medium *_medium;
        friend class Medium;

    public:
        [[nodiscard]] auto priority() const noexcept { return _medium->_priority; }

    public:
        Instance(const Pipeline &pipeline, const Medium *medium) noexcept
            : _pipeline{pipeline}, _medium{medium} {}
        virtual ~Instance() noexcept = default;
        template<typename T = Medium>
            requires std::is_base_of_v<Medium, T>
        [[nodiscard]] auto node() const noexcept { return static_cast<const T *>(_medium); }
        [[nodiscard]] auto &pipeline() const noexcept { return _pipeline; }
        [[nodiscard]] virtual luisa::unique_ptr<Closure> closure(
            Expr<Ray> ray, const SampledWavelengths &swl, Expr<float> time) const noexcept = 0;
    };

protected:
    [[nodiscard]] virtual luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept = 0;

public:
    Medium(Scene *scene, const SceneNodeDesc *desc) noexcept;
    [[nodiscard]] virtual bool is_null() const noexcept { return false; }
    [[nodiscard]] virtual bool is_vacuum() const noexcept { return false; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept;
};

}// namespace luisa::render