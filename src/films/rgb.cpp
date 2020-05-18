//
// Created by Mike Smith on 2020/2/2.
//

#include <OpenImageIO/imageio.h>
#include <core/film.h>

namespace luisa {

class RGBFilm : public Film {

private:
    std::unique_ptr<Kernel> _postprocess_kernel;

public:
    RGBFilm(Device *device, const ParameterSet &parameters);
    void postprocess(KernelDispatcher &dispatch) override;
    void save(std::string_view filename) override;
};

void RGBFilm::postprocess(KernelDispatcher &dispatch) {
    auto pixel_count = _resolution.x * _resolution.y;
    dispatch(*_postprocess_kernel, pixel_count, [&](KernelArgumentEncoder &encode) {
        encode("accumulation_buffer", *_accumulation_buffer);
        encode("pixel_count", pixel_count);
    });
    _accumulation_buffer->synchronize(dispatch);
}

void RGBFilm::save(std::string_view filename) {
    auto path = _device->context().working_path(filename);
    auto image = OIIO::ImageOutput::create(path.string());
    LUISA_EXCEPTION_IF(image == nullptr, "Failed to create output image: ", path);
    OIIO::ImageSpec image_spec{static_cast<int>(_resolution.x), static_cast<int>(_resolution.y), 4, OIIO::TypeDesc::FLOAT};
    LUISA_INFO("Saving film: ", path);
    image->open(path.string(), image_spec);
    image->write_image(OIIO::TypeDesc::FLOAT, _accumulation_buffer->data());
    image->close();
}

RGBFilm::RGBFilm(Device *device, const ParameterSet &parameters) : Film{device, parameters} {
    _postprocess_kernel = device->load_kernel("film::rgb::postprocess");
}

}

LUISA_EXPORT_PLUGIN_CREATOR(luisa::RGBFilm)
