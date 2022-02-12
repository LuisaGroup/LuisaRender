//
// Created by Mike Smith on 2022/1/12.
//

#include <luisa-compute.h>
#include <base/shape.h>

namespace luisa::render {

namespace detail {

class FakePoint final : public Shape {

private:
    [[nodiscard]] static auto _default_desc() noexcept {
        static auto desc = [] {
            static SceneNodeDesc d{"__fakepoint_default_desc", SceneNodeTag::SHAPE};
            d.define(SceneNodeTag::SHAPE, "fakepoint", {});
            return &d;
        }();
        return desc;
    }

public:
    explicit FakePoint(Scene *scene) noexcept : Shape{scene, _default_desc()} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] luisa::span<const float3> positions() const noexcept override {
        static const std::array p{
            make_float3(0.0f, 0.0f, 0.0f),
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f)};
        return p;
    }
    [[nodiscard]] luisa::span<const Shape::VertexAttribute> attributes() const noexcept override {
        static const auto attr = Shape::VertexAttribute::encode(
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(1.0f, 0.0f, 0.0f),
            make_float2(0.0f));
        static const std::array a{attr, attr, attr};
        return a;
    }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override {
        static const Triangle t{0u, 1u, 2u};
        return {&t, 1u};
    }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override { return {}; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
    [[nodiscard]] bool is_virtual() const noexcept override { return true; }
};

}// namespace detail

class VirtualShape final : public Shape {

private:
    detail::FakePoint _point;
    std::array<const Shape *, 1u> _children;

public:
    VirtualShape(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Shape{scene, desc}, _point{scene}, _children{&_point} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_mesh() const noexcept override { return false; }
    [[nodiscard]] bool is_virtual() const noexcept override { return true; }
    [[nodiscard]] bool deformable() const noexcept override { return false; }
    [[nodiscard]] luisa::span<const float3> positions() const noexcept override { return {}; }
    [[nodiscard]] luisa::span<const Shape::VertexAttribute> attributes() const noexcept override { return {}; }
    [[nodiscard]] luisa::span<const Triangle> triangles() const noexcept override { return {}; }
    [[nodiscard]] luisa::span<const Shape *const> children() const noexcept override { return _children; }
};

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::VirtualShape)
