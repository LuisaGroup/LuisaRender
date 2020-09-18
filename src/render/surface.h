//
// Created by Mike Smith on 2020/9/17.
//

#pragma once

#include <compute/dsl.h>

namespace luisa::render {

struct DataBlock {
    float4 padding;
};

}

LUISA_STRUCT(luisa::render::DataBlock, padding)

namespace luisa::render {

using compute::dsl::Expr;

struct SurfaceEvaluation {
    Expr<float3> emission;
    Expr<float3> bsdf;
    Expr<float> pdf_bsdf;
    Expr<float3> sampled_wi;
    Expr<float3> sampled_bsdf;
    Expr<float> sampled_pdf_bsdf;
};

class SurfaceShader {

private:
    [[nodiscard]] virtual SurfaceEvaluation _evaluate(Expr<float2> uv, Expr<float3> wo, Expr<float3> wi, Expr<float2> u2, Expr<DataBlock> data_ref) const = 0;
    
    [[nodiscard]] virtual Expr<float3> _emission(Expr<float2> uv, Expr<float3> wo, Expr<DataBlock> data_ref) const {
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
    [[nodiscard]] SurfaceEvaluation evaluate(Expr<float2> uv, Expr<float3> wo, Expr<float3> wi, Expr<float2> u2, Expr<DataBlock> data_ref) const {
        return _evaluate(uv, wo, wi, u2, data_ref);
    }
    
    [[nodiscard]] Expr<float3> emission(Expr<float2> uv, Expr<float3> wo, Expr<DataBlock> data_ref) const {
        return _emission(uv, wo, data_ref);
    }
    
    [[nodiscard]] uint required_data_block_count() const noexcept { return _required_data_block_count(); }
    [[nodiscard]] uint type_uid() const noexcept { return _type_uid(); }
    [[nodiscard]] bool is_emissive() const noexcept { return _is_emissive(); }
    
    void encode_data(DataBlock *blocks) { _encode_data(blocks); }
};

template<typename Impl>
class Surface : public SurfaceShader {
    
    [[nodiscard]] SurfaceEvaluation _evaluate(Expr<float2> uv, Expr<float3> wo, Expr<float3> wi, Expr<float2> u2, Expr<DataBlock> data_ref) const final {
        return Impl::evaluate(uv, wo, wi, u2, compute::dsl::reinterpret<typename Impl::Data>(data_ref));
    }
    
    [[nodiscard]] Expr<float3> _emission(Expr<float2> uv, Expr<float3> wo, Expr<DataBlock> data_ref) const final {
        if constexpr (Impl::is_emissive) {
            LUISA_EXCEPTION("Invalid emission evaluation on non-emissive surface shader.");
        } else {
            return Impl::emission(uv, wo, compute::dsl::reinterpret<typename Impl::Data>(data_ref));
        }
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
