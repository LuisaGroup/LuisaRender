//
// Created by Mike Smith on 2020/9/17.
//

#pragma once

#include <optional>
#include <render/data_block.h>

namespace luisa::render {

struct Emission {
    float3 L;
};

struct BSDFEvaluation {
    float3 f;
    float pdf;
};

struct BSDFSample {
    float3 wi;
    float3 f;
    float pdf;
};

struct Scattering {
    Emission emission;
    BSDFEvaluation evaluation;
    BSDFSample sample;
};

}

LUISA_STRUCT(luisa::render::Emission, L)
LUISA_STRUCT(luisa::render::BSDFEvaluation, f, pdf)
LUISA_STRUCT(luisa::render::BSDFSample, wi, f, pdf)
LUISA_STRUCT(luisa::render::Scattering, emission, evaluation, sample)

namespace luisa::render {

using compute::dsl::Var;
using compute::dsl::Expr;

class SurfaceShader {

public:
    enum EvaluateComponent : uint {
        EVAL_EMISSION = 1u,
        EVAL_BSDF = 1u << 1u,
        EVAL_BSDF_SAMPLING = 1u << 2u,
        EVAL_ALL = 0xffffffffu
    };

private:
    [[nodiscard]] virtual Expr<Scattering> _evaluate(Expr<float2> uv, Expr<float3> n, Expr<float3> wo, Expr<float3> wi, Expr<float2> u2, Expr<DataBlock> data_ref, uint comp) const = 0;
    [[nodiscard]] virtual Expr<Emission> _emission(Expr<float2> uv, Expr<float3> n, Expr<float3> w, Expr<DataBlock> data_ref) const {
        LUISA_EXCEPTION("Invalid sampling operation on non-emissive surface.");
    }
    
    [[nodiscard]] virtual uint _required_data_block_count() const noexcept = 0;
    
    [[nodiscard]] virtual bool _is_emissive() const noexcept = 0;
    
    [[nodiscard]] virtual uint _type_uid() const noexcept = 0;
    
    virtual void _encode_data(DataBlock *blocks) const = 0;

protected:
    [[nodiscard]] static uint _assign_uid() noexcept {
        static auto uid_counter = 0u;
        return ++uid_counter;
    }

public:
    virtual ~SurfaceShader() noexcept = default;
    
    [[nodiscard]] Expr<Scattering> evaluate(Var<float2> uv, Var<float3> n, Var<float3> wo, Var<float3> wi, Var<float2> u2, Expr<DataBlock> data_ref, uint comp = EVAL_ALL) const {
        return _evaluate(uv, n, wo, wi, u2, data_ref, comp);
    }
    
    [[nodiscard]] Expr<Emission> emission(Var<float2> uv, Var<float3> n, Var<float3> wo, Expr<DataBlock> data_ref) const {
        return _emission(uv, n, wo, data_ref);
    }
    
    [[nodiscard]] uint required_data_block_count() const noexcept { return _required_data_block_count(); }
    [[nodiscard]] uint type_uid() const noexcept { return _type_uid(); }
    [[nodiscard]] bool is_emissive() const noexcept { return _is_emissive(); }
    
    void encode_data(DataBlock *blocks) { _encode_data(blocks); }
};

template<typename Impl>
class Surface : public SurfaceShader {
    
    [[nodiscard]] Expr<Scattering> _evaluate(Expr<float2> uv, Expr<float3> n, Expr<float3> wo, Expr<float3> wi, Expr<float2> u2, Expr<DataBlock> data_ref, uint comp) const final {
        Var data = compute::dsl::reinterpret<typename Impl::Data>(data_ref);
        return Impl::evaluate(uv, n, wo, wi, u2, data, comp);
    }
    
    [[nodiscard]] Expr<Emission> _emission(Expr<float2> uv, Expr<float3> n, Expr<float3> wo, Expr<DataBlock> data_ref) const final {
        // Not using static-if's, to make MSVC happy...
        return _emission_impl(static_cast<const Impl *>(this), uv, n, wo, data_ref);
    }
    
    template<typename I, std::enable_if_t<I::is_emissive, int> = 0>
    [[nodiscard]] static Expr<Emission> _emission_impl(const I *, Expr<float2> uv, Expr<float3> n, Expr<float3> wo, Expr<DataBlock> data_ref) {
        Var data = compute::dsl::reinterpret<typename I::Data>(data_ref);
        return I::emission(uv, n, wo, data);
    }
    
    template<typename I, std::enable_if_t<!I::is_emissive, int> = 0>
    [[noreturn]] static Expr<Emission> _emission_impl(const I *, Expr<float2>, Expr<float3>, Expr<float3>, Expr<DataBlock>) {
        LUISA_EXCEPTION("Invalid emission evaluation on non-emissive surface shader.");
    }
    
    [[nodiscard]] uint _required_data_block_count() const noexcept final {
        return (sizeof(typename Impl::Data) + sizeof(DataBlock) - 1u) / sizeof(DataBlock);
    }
    
    [[nodiscard]] bool _is_emissive() const noexcept final {
        return Impl::is_emissive;
    }
    
    [[nodiscard]] uint _type_uid() const noexcept final {
        static uint uid = SurfaceShader::_assign_uid();
        return uid;
    }
    
    void _encode_data(DataBlock *blocks) const final {
        *reinterpret_cast<typename Impl::Data *>(blocks) = static_cast<const Impl *>(this)->data();
    }
    
};

}
