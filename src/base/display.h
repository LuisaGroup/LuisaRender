//
// Created by Mike Smith on 2022/11/9.
//

#include <base/film.h>
#include <gui/window.h>
#include <gui/framerate.h>

namespace luisa::render {

using compute::Framerate;
using compute::Image;
using compute::Buffer;
using compute::Shader1D;
using compute::Window;

class Display {

public:
    enum struct ToneMapper : uint {
        NONE,
        ACES,
        UNCHARTED2,
    };

private:
    luisa::string _name;
    luisa::unique_ptr<Window> _window;
    Image<float> _converted;
    luisa::vector<std::array<uint8_t, 4u>> _pixels;
    Framerate _framerate;
    Clock _clock;
    uint _last_spp{};
    std::shared_future<Shader1D<uint /* tone mapping method */, float /* exposure */>> _convert;
    ToneMapper _tone_mapper;
    float _exposure{};

public:
    explicit Display(luisa::string name) noexcept;
    void reset(CommandBuffer &command_buffer, const Film::Instance *film) noexcept;
    [[nodiscard]] bool should_close() const noexcept;
    bool update(CommandBuffer &command_buffer, uint spp) noexcept;
    bool idle() noexcept;
};

}// namespace luisa::render
