//
// Created by Mike Smith on 2023/6/14.
//

#include <runtime/image.h>
#include <runtime/swapchain.h>

#include <gui/window.h>

#include <base/film.h>
#include <base/scene.h>
#include <base/pipeline.h>

namespace luisa::render {

class Display final : public Film {

public:
    enum struct ToneMapping : uint8_t {
        NONE,
        UNCHARTED2,
        ACES
    };

private:
    Film *_base;
    float _target_fps;
    float _exposure;
    uint8_t _back_buffers;
    ToneMapping _tone_mapping;
    bool _hdr;
    bool _vsync;

public:
    Display(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Film{scene, desc},
          _base{scene->load_film(desc->property_node("base"))},
          _target_fps{std::clamp(desc->property_float_or_default("target_fps", 30.f), 1.f, 1024.f)},
          _back_buffers{static_cast<uint8_t>(std::clamp(
              desc->property_uint_or_default("back_buffers", 3u), 1u, 8u))},
          _exposure{std::clamp(
              desc->property_float_or_default(
                  "exposure", lazy_construct([desc]() noexcept {
                      return desc->property_float_or_default("exposure", 0.0f);
                  })),
              -10.f, 10.f)},
          _tone_mapping{[desc] {
              auto tm = desc->property_string_or_default(
                  "tone_mapping", lazy_construct([desc]() noexcept {
                      return desc->property_string_or_default(
                          "tonemapping", "none");
                  }));
              for (auto &c : tm) { c = static_cast<char>(std::tolower(c)); }
              if (tm == "uncharted2") { return ToneMapping::UNCHARTED2; }
              if (tm == "aces") { return ToneMapping::ACES; }
              if (tm != "none") {
                  LUISA_WARNING_WITH_LOCATION(
                      "Unknown tone mapping operator: \"{}\". "
                      "Available options are: \"none\", \"uncharted2\", \"aces\".",
                      tm);
              }
              return ToneMapping::NONE;
          }()},
          _hdr{desc->property_bool_or_default(
              "HDR", lazy_construct([desc]() noexcept {
                  return desc->property_bool_or_default("hdr", false);
              }))},
          _vsync{desc->property_bool_or_default(
              "VSync", lazy_construct([desc]() noexcept {
                  return desc->property_bool_or_default(
                      "vsync", lazy_construct([desc]() noexcept {
                          return desc->property_bool_or_default("vertical_sync", true);
                      }));
              }))} {}

    [[nodiscard]] luisa::string_view impl_type() const noexcept override {
        return LUISA_RENDER_PLUGIN_NAME;
    }

    [[nodiscard]] uint2 resolution() const noexcept override { return _base->resolution(); }
    [[nodiscard]] float clamp() const noexcept override { return _base->clamp(); }
    [[nodiscard]] auto target_fps() const noexcept { return _target_fps; }
    [[nodiscard]] auto hdr() const noexcept { return _hdr; }
    [[nodiscard]] auto vsync() const noexcept { return _vsync; }
    [[nodiscard]] auto back_buffers() const noexcept { return _back_buffers; }
    [[nodiscard]] auto exposure() const noexcept { return _exposure; }
    [[nodiscard]] auto tone_mapping() const noexcept { return _tone_mapping; }

    [[nodiscard]] luisa::unique_ptr<Instance> build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

using namespace compute;

class DisplayInstance final : public Film::Instance {

private:
    luisa::unique_ptr<Film::Instance> _base;
    luisa::unique_ptr<Window> _window;
    Image<float> _framebuffer;
    Swapchain _swapchain;
    Shader2D<> _blit;
    Clock _clock;
    mutable double _last_frame_time;

private:
    [[nodiscard]] auto _tone_mapping_uncharted2(Expr<float3> color) noexcept {
        static constexpr auto a = 0.15f;
        static constexpr auto b = 0.50f;
        static constexpr auto c = 0.10f;
        static constexpr auto d = 0.20f;
        static constexpr auto e = 0.02f;
        static constexpr auto f = 0.30f;
        static constexpr auto white = 11.2f;
        auto op = [](auto x) noexcept {
            return (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f;
        };
        return op(1.6f * color) / op(white);
    }
    [[nodiscard]] auto _tone_mapping_aces(Expr<float3> color) noexcept {
        constexpr auto a = 2.51f;
        constexpr auto b = 0.03f;
        constexpr auto c = 2.43f;
        constexpr auto d = 0.59f;
        constexpr auto e = 0.14f;
        return (color * (a * color + b)) / (color * (c * color + d) + e);
    }
    [[nodiscard]] auto _linear_to_srgb(Expr<float3> color) noexcept {
        return ite(color <= .0031308f,
                   color * 12.92f,
                   1.055f * pow(color, 1.f / 2.4f) - .055f);
    }

public:
    DisplayInstance(const Pipeline &pipeline, const Film *film,
                    luisa::unique_ptr<Film::Instance> base) noexcept
        : Film::Instance{pipeline, film},
          _base{std::move(base)} {}

    [[nodiscard]] Film::Accumulation read(Expr<uint2> pixel) const noexcept override {
        return _base->read(pixel);
    }

    void prepare(CommandBuffer &command_buffer) noexcept override {
        _base->prepare(command_buffer);
        auto &&device = pipeline().device();
        auto size = node()->resolution();
        if (!_window) {
            _window = luisa::make_unique<Window>("Display", size);
            auto d = node<Display>();
            _swapchain = device.create_swapchain(
                _window->native_handle(), *command_buffer.stream(),
                size, d->hdr(), d->vsync(), d->back_buffers());
            _framebuffer = device.create_image<float>(
                _swapchain.backend_storage(), size);
            _blit = device.compile<2>([&] {
                auto p = dispatch_id().xy();
                auto color = _base->read(p).average * std::exp2(d->exposure());
                switch (d->tone_mapping()) {
                    case Display::ToneMapping::NONE: break;
                    case Display::ToneMapping::UNCHARTED2: color = _tone_mapping_uncharted2(color); break;
                    case Display::ToneMapping::ACES: color = _tone_mapping_aces(color); break;
                }
                if (_framebuffer.storage() == PixelStorage::BYTE4) {// LDR
                    color = _linear_to_srgb(color);
                }
                _framebuffer->write(p, make_float4(color, 1.f));
            });
        }
        _last_frame_time = _clock.toc();
    }

    void clear(CommandBuffer &command_buffer) noexcept override {
        _base->clear(command_buffer);
    }

    void download(CommandBuffer &command_buffer, float4 *framebuffer) const noexcept override {
        _base->download(command_buffer, framebuffer);
    }

    void release() noexcept override {
        while (_window && !_window->should_close()) {
            _window->poll_events();
        }
        _framebuffer = {};
        _swapchain = {};
        _window.reset();
        _base->release();
    }

    bool show(CommandBuffer &command_buffer) const noexcept override {
        auto interval = 1. / node<Display>()->target_fps();
        if (auto current_time = _clock.toc();
            current_time - _last_frame_time >= interval) {
            _last_frame_time = current_time;
            _window->poll_events();
            if (_window->should_close()) {
                command_buffer << synchronize();
                exit(0);// FIXME: exit gracefully
            }
            command_buffer << _blit().dispatch(node()->resolution())
                           << _swapchain.present(_framebuffer);
            return true;
        }
        return false;
    }

protected:
    void _accumulate(Expr<uint2> pixel, Expr<float3> rgb, Expr<float> effective_spp) const noexcept override {
        _base->accumulate(pixel, rgb, effective_spp);
    }
};

luisa::unique_ptr<Film::Instance> Display::build(Pipeline &pipeline,
                                                 CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<DisplayInstance>(
        pipeline, this, _base->build(pipeline, command_buffer));
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::Display)
