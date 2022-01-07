//
// Created by Mike on 2022/1/7.
//

#include <scene/film.h>

namespace luisa::render {

class ColorFilm final : public Film {

public:
    ColorFilm(Scene *scene, const SceneNodeDesc *desc) noexcept : Film{scene, desc} {}
    [[nodiscard]] luisa::unique_ptr<Instance> build(Stream &stream, Pipeline &pipeline) const noexcept override {
        return nullptr;
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "color"; }
};

}

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::ColorFilm)
