//
// Created by Mike on 2021/12/8.
//

#include <base/camera.h>

namespace luisa::render {

Camera &Camera::set_film(Film *film) noexcept {
    _film = film;
    _on_set_film();
    return *this;
}

Camera &Camera::set_filter(Filter *filter) noexcept {
    _filter = filter;
    _on_set_filter();
    return *this;
}

Camera &Camera::set_transform(Transform *transform) noexcept {
    _transform = transform;
    _on_set_transform();
    return *this;
}

}
