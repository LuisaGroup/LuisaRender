//
// Created by Mike on 2021/12/8.
//

#include <core/logging.h>
#include <util/sampling.h>
#include <base/filter.h>

namespace luisa::render {

Filter::Filter(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::FILTER},
      _radius{std::max(desc->property_float_or_default("radius", 0.5f), 1e-3f)},
      _shift{desc->property_float2_or_default(
          "shift", lazy_construct([desc] {
              return make_float2(desc->property_float_or_default("shift", 0.f));
          }))} {}

luisa::unique_ptr<Filter::Instance> Filter::build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<Instance>(pipeline, this);
}

Filter::Instance::Instance(const Pipeline &pipeline, const Filter *filter) noexcept
    : _pipeline{pipeline}, _filter{filter} {
    static constexpr auto n = Filter::look_up_table_size - 1u;
    static constexpr auto inv_n = 1.0f / static_cast<float>(n);
    std::array<float, n> abs_f{};
    _lut[0u] = filter->evaluate(-filter->radius());
    auto integral = 0.0f;
    for (auto i = 0u; i < n; i++) {
        auto x = static_cast<float>(i + 1u) * inv_n * 2.0f - 1.0f;
        _lut[i + 1u] = filter->evaluate(x * filter->radius());
        auto f_mid = 0.5f * (_lut[i] + _lut[i + 1u]);
        integral += f_mid;
        abs_f[i] = std::abs(f_mid);
    }
    auto inv_integral = 1.0f / integral;
    for (auto &f : _lut) { f *= inv_integral; }
    auto [alias_table, pdf] = create_alias_table(abs_f);
    assert(alias_table.size() == n && pdf.size() == n);
    for (auto i = 0u; i < n; i++) {
        _pdf[i] = pdf[i];
        _alias_probs[i] = alias_table[i].prob;
        _alias_indices[i] = alias_table[i].alias;
    }
}

Filter::Sample Filter::Instance::sample(Expr<float2> u) const noexcept {
    using namespace luisa::compute;
    Constant lut = look_up_table();
    Constant pdfs = pdf_table();
    Constant alias_indices = alias_table_indices();
    Constant alias_probs = alias_table_probabilities();
    auto n = look_up_table_size - 1u;
    auto [iy, uy] = sample_alias_table(alias_probs, alias_indices, n, u.x);
    auto [ix, ux] = sample_alias_table(alias_probs, alias_indices, n, u.y);
    auto pdf = pdfs[iy] * pdfs[ix];
    auto f = lerp(lut[ix], lut[ix + 1u], ux) * lerp(lut[iy], lut[iy + 1u], uy);
    auto p = make_float2(make_uint2(ix, iy)) + make_float2(ux, uy);
    auto inv_size = 1.0f / static_cast<float>(look_up_table_size);
    auto pixel = (p * inv_size * 2.0f - 1.0f) * _filter->radius();
    return {pixel + node()->shift(), f / pdf};
}

}// namespace luisa::render
