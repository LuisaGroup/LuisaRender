//
// Created by Mike Smith on 2022/11/9.
//

#include <dsl/syntax.h>
#include <runtime/command_buffer.h>
#include <base/display.h>
#include <base/pipeline.h>

namespace luisa::render {

Display::Display(const char *name, uint2 initial_resolution) noexcept
    : _window{luisa::make_unique<Window>(name, initial_resolution)},
      _tone_mapper{ToneMapper::UNCHARTED2},
      _exposure{0.f} {}

void Display::reset(CommandBuffer &command_buffer, const Film::Instance *film) noexcept {
    using namespace luisa::compute;
    auto &&device = film->pipeline().device();
    auto resolution = film->node()->resolution();
    _window->set_size(resolution);
    _pixels.resize(resolution.x * resolution.y);
    _converted = device.create_image<float>(PixelStorage::BYTE4, resolution);
    _convert = device.compile_async<1>([film, w = resolution.x, this](UInt tone_mapper, Float exposure) noexcept {
        auto p = make_uint2(dispatch_x() % w, dispatch_x() / w);
        auto x = film->read(p).average * pow(2.f, exposure);
        $switch(tone_mapper) {
            $case(luisa::to_underlying(ToneMapper::NONE)){/* do nothing */};
            $case(luisa::to_underlying(ToneMapper::ACES)) {
                constexpr auto a = 2.51f;
                constexpr auto b = 0.03f;
                constexpr auto c = 2.43f;
                constexpr auto d = 0.59f;
                constexpr auto e = 0.14f;
                x = x * (a * x + b) / (x * (c * x + d) + e);
            };
            $case(luisa::to_underlying(ToneMapper::UNCHARTED2)) {
                constexpr auto F = [](auto x) noexcept {
                    constexpr auto A = 0.22f;
                    constexpr auto B = 0.30f;
                    constexpr auto C = 0.10f;
                    constexpr auto D = 0.20f;
                    constexpr auto E = 0.01f;
                    constexpr auto F = 0.30f;
                    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
                };
                constexpr auto WHITE = 11.2f;
                x = F(1.6f * x) / F(WHITE);
            };
            $default { unreachable(); };
        };
        // linear to sRGB
        x = ite(x <= .0031308f, 12.92f * x, 1.055f * pow(x, 1.f / 2.4f) - .055f);
        _converted.write(p, make_float4(x, 1.f));
    });
}

void Display::update(CommandBuffer &command_buffer, uint spp) noexcept {
    if (_window->should_close()) { return; }
    command_buffer << _convert.get()(luisa::to_underlying(_tone_mapper), _exposure)
                          .dispatch(_pixels.size())
                   << _converted.copy_to(_pixels.data())
                   << compute::synchronize();
    _framerate.record(spp - _last_spp);
    _last_spp = spp;
    _window->run_one_frame([&] {
        _window->set_background(_pixels.data(), _converted.size());
        ImGui::Begin("Console", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Frame: %u", spp);
        ImGui::Text("Time: %.1fs", _clock.toc() * 1e-3);
        ImGui::Text("FPS: %.2f", _framerate.report());
        static constexpr std::array tone_mapper_names{
            "None",
            "ACES",
            "Uncharted2",
        };
        ImGui::Text("Tone Mapper");
        for (auto i = 0u; i < tone_mapper_names.size(); i++) {
            ImGui::SameLine();
            if (ImGui::RadioButton(tone_mapper_names[i], luisa::to_underlying(_tone_mapper) == i)) {
                _tone_mapper = static_cast<ToneMapper>(i);
            }
        }
        ImGui::SliderFloat("Exposure", &_exposure, -10.f, 10.f, "%.1f");
        ImGui::End();
    });
}

void Display::idle() noexcept {
    if (!_window->should_close()) {
        _window->run_one_frame([] {});
    }
}

}// namespace luisa::render
