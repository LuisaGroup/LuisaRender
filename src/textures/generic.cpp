//
// Created by Mike Smith on 2022/1/28.
//

#include <core/clock.h>
#include <core/thread_pool.h>
#include <util/imageio.h>
#include <util/half.h>
#include <base/texture.h>
#include <base/pipeline.h>

namespace luisa::render {

using namespace luisa::compute;

class GenericTexture final : public ImageTexture {

private:
    std::shared_future<LoadedImage> _img;

private:
    [[nodiscard]] const LoadedImage &_image() const noexcept override { return _img.get(); }

public:
    GenericTexture(Scene *scene, const SceneNodeDesc *desc) noexcept
        : ImageTexture{scene, desc} {
        auto path = desc->property_path("file");
        _img = ThreadPool::global().async([path = std::move(path), sloc = desc->source_location()] {
            auto image = LoadedImage::load(path);
            if (auto s = image.pixel_storage();
                s == PixelStorage::INT1 ||
                s == PixelStorage::INT2 ||
                s == PixelStorage::INT4) [[unlikely]] {
                LUISA_ERROR(
                    "Texture '{}' with INT{} storage is "
                    "not supported in GenericTexture. [{}]",
                    path.string(), pixel_storage_channel_count(s),
                    sloc.string());
            }
            return image;
        });
    }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "generic"; }
    [[nodiscard]] Category category() const noexcept override { return Category::GENERIC; }
    [[nodiscard]] bool is_black() const noexcept override { return false; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::GenericTexture)
