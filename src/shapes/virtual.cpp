//
// Created by Mike Smith on 2022/1/12.
//

#include <luisa-compute.h>
#include <scene/shape.h>

namespace luisa::render {

namespace detail {

class FakePoint final : public Shape {

private:
    [[nodiscard]] static auto _default_desc() noexcept {
        static auto desc = []{
            static SceneNodeDesc d{"__fakepoint_default_desc", SceneNodeTag::SHAPE};
            d.define(SceneNodeTag::SHAPE, "fakepoint", {});
            return &d;
        }();
        return desc;
    }

public:
    FakePoint() noexcept : Shape{nullptr, _default_desc()} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "fakepoint"; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] luisa::span<const float3> positions() const noexcept override {
        static const auto p = make_float3(0.0f);
        return {&p, 1u};
    }
    [[nodiscard]] luisa::span<const VertexAttribute> attributes() const noexcept override {
        static const auto a = VertexAttribute::encode(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(1.0f, 0.0f, 0.0f),
            make_float2(0.0f));
        return {&a, 1u};
    }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override {
        static const Triangle t{0u, 0u, 0u};
        return {&t, 1u};
    }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override { return {}; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
    [[nodiscard]] bool is_virtual() const noexcept override { return true; }

public:
    [[nodiscard]] static auto instance() noexcept {
        static FakePoint p;
        return &p;
    }
};

}

class VirtualShape final : public Shape {

public:
    VirtualShape(Scene *scene, const SceneNodeDesc *desc) noexcept : Shape{scene, desc} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return "virtual"; }
    [[nodiscard]] bool is_mesh() const noexcept override { return false; }
    [[nodiscard]] bool is_virtual() const noexcept override { return true; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
    [[nodiscard]] luisa::span<const float3> positions() const noexcept override { return {}; }
    [[nodiscard]] luisa::span<const VertexAttribute> attributes() const noexcept override { return {}; }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override { return {}; }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override {
        static const std::array<const Shape *, 1u> shapes{detail::FakePoint::instance()};
        return shapes;
    }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::VirtualShape)
