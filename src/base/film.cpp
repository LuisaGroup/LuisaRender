//
// Created by Mike on 2021/12/14.
//

#include <dsl/sugar.h>
#include <base/film.h>

namespace luisa::render {

Film::Film(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::FILM} {}

void Film::Instance::accumulate(Expr<uint2> pixel, Expr<float3> rgb,
                                Expr<float> effective_spp) const noexcept {
    $outline {
#ifndef NDEBUG
        $if(all(pixel >= 0u && pixel < node()->resolution())) {
#endif
            _accumulate(pixel, rgb, effective_spp);
#ifndef NDEBUG
        };
#endif
    };
}

}// namespace luisa::render
