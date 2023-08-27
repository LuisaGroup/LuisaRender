//
// Created by Mike on 2021/12/8.
//

#include <random>
#include <numeric>

#include <sdl/scene_node_desc.h>
#include <base/filter.h>
#include <base/scene.h>
#include <base/camera.h>
#include <base/pipeline.h>

namespace luisa::render {

Camera::Camera(Scene *scene, const SceneNodeDesc *desc) noexcept
    : SceneNode{scene, desc, SceneNodeTag::CAMERA},
      _film{scene->load_film(desc->property_node("film"))},
      _filter{scene->load_filter(desc->property_node_or_default(
          "filter", SceneNodeDesc::shared_default_filter("Box")))},
      _transform{scene->load_transform(desc->property_node_or_default("transform"))},
      _shutter_span{desc->property_float2_or_default(
          "shutter_span", lazy_construct([desc] {
              return make_float2(desc->property_float_or_default(
                  "shutter_span", 0.0f));
          }))},
      _shutter_samples{desc->property_uint_or_default("shutter_samples", 0u)},// 0 means default
      _spp{desc->property_uint_or_default("spp", 1024u)} {

    // For compatibility with older scene description versions
    if (_transform == nullptr) {
        static constexpr auto default_position = make_float3(0.f, 0.f, 0.f);
        static constexpr auto default_front = make_float3(0.f, 0.f, -1.f);
        static constexpr auto default_up = make_float3(0.f, 1.f, 0.f);
        auto position = desc->property_float3_or_default("position", default_position);
        auto front = desc->property_float3_or_default(
            "front", lazy_construct([desc, position] {
                auto look_at = desc->property_float3_or_default("look_at", position + default_front);
                return normalize(look_at - position);
            }));
        auto up = desc->property_float3_or_default("up", default_up);
        if (!all(position == default_position && front == default_front && up == default_up)) {
            SceneNodeDesc d{luisa::format("{}$transform", desc->identifier()), SceneNodeTag::TRANSFORM};
            d.define(SceneNodeTag::TRANSFORM, "View", desc->source_location());
            d.add_property("position", SceneNodeDesc::number_list{position.x, position.y, position.z});
            d.add_property("front", SceneNodeDesc::number_list{front.x, front.y, front.z});
            d.add_property("up", SceneNodeDesc::number_list{up.x, up.y, up.z});
            _transform = scene->load_transform(&d);
        }
    }

    if (_shutter_span.y < _shutter_span.x) [[unlikely]] {
        LUISA_ERROR(
            "Invalid time span: [{}, {}]. [{}]",
            _shutter_span.x, _shutter_span.y,
            desc->source_location().string());
    }
    if (_shutter_span.x != _shutter_span.y) {
        if (_shutter_samples == 0u) {
            _shutter_samples = std::min(_spp, 256u);
        } else if (_shutter_samples > _spp) {
            LUISA_WARNING(
                "Too many shutter samples ({}), "
                "clamping to samples per pixel ({}). [{}]",
                _shutter_samples, _spp,
                desc->source_location().string());
            _shutter_samples = _spp;
        }
        auto shutter_time_points = desc->property_float_list_or_default("shutter_time_points");
        auto shutter_weights = desc->property_float_list_or_default("shutter_weights");
        if (shutter_time_points.size() != shutter_weights.size()) [[unlikely]] {
            LUISA_ERROR(
                "Number of shutter time points and "
                "number of shutter weights mismatch. [{}]",
                desc->source_location().string());
        }
        if (std::any_of(shutter_weights.cbegin(), shutter_weights.cend(), [](auto w) noexcept {
                return w < 0.0f;
            })) [[unlikely]] {
            LUISA_ERROR(
                "Found negative shutter weight. [{}]",
                desc->source_location().string());
        }
        if (shutter_time_points.empty()) {
            _shutter_points.emplace_back(ShutterPoint{_shutter_span.x, 1.0f});
            _shutter_points.emplace_back(ShutterPoint{_shutter_span.y, 1.0f});
        } else {
            luisa::vector<uint> indices(shutter_time_points.size());
            std::iota(indices.begin(), indices.end(), 0u);
            if (auto iter = std::remove_if(indices.begin(), indices.end(), [&](auto i) noexcept {
                    auto t = shutter_time_points[i];
                    return t < _shutter_span.x || t > _shutter_span.y;
                });
                iter != indices.end()) [[unlikely]] {
                LUISA_WARNING(
                    "Out-of-shutter samples (count = {}) "
                    "are to be removed. [{}]",
                    std::distance(iter, indices.end()),
                    desc->source_location().string());
                indices.erase(iter, indices.end());
            }
            std::sort(indices.begin(), indices.end(), [&](auto lhs, auto rhs) noexcept {
                return shutter_time_points[lhs] < shutter_time_points[rhs];
            });
            if (auto iter = std::unique(indices.begin(), indices.end(), [&](auto lhs, auto rhs) noexcept {
                    return shutter_time_points[lhs] == shutter_time_points[rhs];
                });
                iter != indices.end()) [[unlikely]] {
                LUISA_WARNING(
                    "Duplicate shutter samples (count = {}) "
                    "are to be removed. [{}]",
                    std::distance(iter, indices.end()),
                    desc->source_location().string());
                indices.erase(iter, indices.end());
            }
            _shutter_points.reserve(indices.size() + 2u);
            _shutter_points.resize(indices.size());
            std::transform(
                indices.cbegin(), indices.cend(),
                std::back_inserter(_shutter_points), [&](auto i) noexcept {
                    return ShutterPoint{shutter_time_points[i], shutter_weights[i]};
                });
            if (!_shutter_points.empty()) {
                if (auto ts = _shutter_span.x; _shutter_points.front().time > ts) {
                    _shutter_points.insert(
                        _shutter_points.begin(),
                        ShutterPoint{ts, _shutter_points.front().weight});
                }
                if (auto te = _shutter_span.y; _shutter_points.back().time < te) {
                    _shutter_points.emplace_back(
                        ShutterPoint{te, _shutter_points.back().weight});
                }
            }
        }
    }

    // render file
    _file = desc->property_path_or_default(
        "file", std::filesystem::canonical(
                    desc->source_location() ?
                        desc->source_location().file()->parent_path() :
                        std::filesystem::current_path()) /
                    "render.exr");
    if (auto folder = _file.parent_path();
        !std::filesystem::exists(folder)) {
        std::filesystem::create_directories(folder);
    }
}

auto Camera::shutter_weight(float time) const noexcept -> float {
    if (time < _shutter_span.x || time > _shutter_span.y) { return 0.0f; }
    if (_shutter_span.x == _shutter_span.y) { return 1.0f; }
    auto ub = std::upper_bound(
        _shutter_points.cbegin(), _shutter_points.cend(), time,
        [](auto lhs, auto rhs) noexcept { return lhs < rhs.time; });
    auto u = std::distance(_shutter_points.cbegin(), ub);
    auto p0 = _shutter_points[u - 1u];
    auto p1 = _shutter_points[u];
    auto t = (time - p0.time) / (p1.time - p0.time);
    return std::lerp(p0.weight, p1.weight, t);
}

auto Camera::shutter_samples() const noexcept -> vector<ShutterSample> {
    if (_shutter_span.x == _shutter_span.y) {
        ShutterPoint sp{_shutter_span.x, 1.0f};
        return {ShutterSample{sp, _spp}};
    }
    auto duration = _shutter_span.y - _shutter_span.x;
    auto inv_n = 1.0f / static_cast<float>(_shutter_samples);
    std::uniform_real_distribution<float> dist{};
    std::default_random_engine random{std::random_device{}()};
    luisa::vector<ShutterSample> buckets(_shutter_samples);
    for (auto bucket = 0u; bucket < _shutter_samples; bucket++) {
        auto ts = static_cast<float>(bucket) * inv_n * duration;
        auto te = static_cast<float>(bucket + 1u) * inv_n * duration;
        auto a = dist(random);
        auto t = std::lerp(ts, te, a);
        auto w = shutter_weight(t);
        buckets[bucket].point = ShutterPoint{t, w};
    }
    luisa::vector<uint> indices(_shutter_samples);
    std::iota(indices.begin(), indices.end(), 0u);
    std::shuffle(indices.begin(), indices.end(), random);
    auto remainder = _spp % _shutter_samples;
    auto samples_per_bucket = _spp / _shutter_samples;
    for (auto i = 0u; i < remainder; i++) { buckets[indices[i]].spp = samples_per_bucket + 1u; }
    for (auto i = remainder; i < _shutter_samples; i++) { buckets[indices[i]].spp = samples_per_bucket; }
    auto sum_weights = std::accumulate(buckets.cbegin(), buckets.cend(), 0.0, [](auto lhs, auto rhs) noexcept {
        return lhs + rhs.point.weight * rhs.spp;
    });
    if (sum_weights == 0.0) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid shutter samples generated. "
            "Falling back to uniform shutter curve.");
        for (auto &s : buckets) { s.point.weight = 1.0f; }
    } else {
        auto scale = _spp / sum_weights;
        for (auto &s : buckets) {
            s.point.weight = static_cast<float>(s.point.weight * scale);
        }
    }
    return buckets;
}

Camera::Instance::Instance(Pipeline &pipeline, CommandBuffer &command_buffer, const Camera *camera) noexcept
    : _pipeline{&pipeline}, _camera{camera},
      _film{camera->film()->build(pipeline, command_buffer)},
      _filter{pipeline.build_filter(command_buffer, camera->filter())} {
    pipeline.register_transform(camera->transform());
}

Camera::Sample Camera::Instance::generate_ray(Expr<uint2> pixel_coord, Expr<float> time,
                                              Expr<float2> u_filter, Expr<float2> u_lens) const noexcept {
    auto [filter_offset, filter_weight] = filter()->sample(u_filter);
    auto pixel = make_float2(pixel_coord) + .5f + filter_offset;
    auto [ray, weight] = _generate_ray_in_camera_space(pixel, u_lens, time);
    weight *= filter_weight;
    auto c2w = camera_to_world();
    auto o = make_float3(c2w * make_float4(ray->origin(), 1.f));
    auto d = normalize(make_float3x3(c2w) * ray->direction());
    ray->set_origin(o);
    ray->set_direction(d);
    return {std::move(ray), pixel, weight};
}

Camera::SampleDifferential Camera::Instance::generate_ray_differential(Expr<uint2> pixel_coord, Expr<float> time,
                                                                       Expr<float2> u_filter, Expr<float2> u_lens) const noexcept {
    auto [filter_offset, filter_weight] = filter()->sample(u_filter);
    auto pixel = make_float2(pixel_coord) + .5f + filter_offset;
    auto [central_ray, central_weight] = _generate_ray_in_camera_space(pixel, u_lens, time);
    auto [x_ray, x_weight] = _generate_ray_in_camera_space(pixel + make_float2(1.f, 0.f), u_lens, time);
    auto [y_ray, y_weight] = _generate_ray_in_camera_space(pixel + make_float2(0.f, 1.f), u_lens, time);
    auto weight = central_weight * filter_weight;
    auto c2w = camera_to_world();
    auto c2w_n = make_float3x3(c2w);
    auto c_o = make_float3(c2w * make_float4(central_ray->origin(), 1.f));
    auto c_d = normalize(c2w_n * central_ray->direction());
    central_ray->set_origin(c_o);
    central_ray->set_direction(c_d);
    auto rx_o = make_float3(c2w * make_float4(x_ray->origin(), 1.f));
    auto rx_d = normalize(c2w_n * x_ray->direction());
    auto ry_o = make_float3(c2w * make_float4(y_ray->origin(), 1.f));
    auto ry_d = normalize(c2w_n * y_ray->direction());
    return {.ray_differential = {
                .ray = std::move(central_ray),
                .rx_origin = rx_o,
                .ry_origin = ry_o,
                .rx_direction = rx_d,
                .ry_direction = ry_d},
            .pixel = pixel,
            .weight = weight};
}

Float4x4 Camera::Instance::camera_to_world() const noexcept {
    return pipeline().transform(node()->transform());
}

}// namespace luisa::render
