//
// Created by Mike Smith on 2020/9/4.
//

#include "geometry.h"

namespace luisa::render {

Geometry::Geometry(Device *device, const std::vector<std::shared_ptr<Shape>> &shapes) {
    
    // calculate memory usage...
    size_t vertex_count = 0u;
    size_t triangle_count = 0u;
    size_t entity_count = 0u;
    size_t instance_count = 0u;
    
    std::queue<Shape *> unvisited_shapes;
    for (auto &&shape: shapes) { unvisited_shapes.emplace(shape.get()); }
    
    std::unordered_set<Shape *> visited_shapes;
    while (!unvisited_shapes.empty()) {
        auto shape = unvisited_shapes.front();
        unvisited_shapes.pop();
        instance_count++;
        if (visited_shapes.count(shape) == 0u) {  // not visited
            entity_count++;
            vertex_count += shape->vertices().size();
            triangle_count += shape->indices().size();
            visited_shapes.emplace(shape);
        }
        for (auto &&child : shape->children()) { unvisited_shapes.emplace(child.get()); }
    }
    
    LUISA_INFO("Creating geometry with ",
               instance_count, " instances, ",
               entity_count, " entities, ",
               triangle_count, " unique triangles and ",
               vertex_count, " unique vertices.");
    
    // allocate buffers
    _positions = device->allocate_buffer<float3>(vertex_count);
    _normals = device->allocate_buffer<float3>(vertex_count);
    _tex_coords = device->allocate_buffer<float2>(vertex_count);
    _triangles = device->allocate_buffer<packed_uint3>(triangle_count);
    _entities = device->allocate_buffer<Entity>(entity_count);
    _instances = device->allocate_buffer<uint>(instance_count);
    _instance_transforms = device->allocate_buffer<float4x4>(instance_count);
    
    // encode shapes
    std::vector<packed_uint3> meshes;
    meshes.reserve(entity_count);
    device->launch(_positions.modify([&](float3 *positions) {
        _normals.modify([&](float3 *normals) {
            _tex_coords.modify([&](float2 *uvs) {
                _triangles.modify([&](packed_uint3 *indices) {
                    _entities.modify([&](Entity *entities) {
                        _instances.modify([&](uint *instances) {
                            _instance_transforms.modify([&](float4x4 *transforms) {
                                _encode(shapes, positions, normals, uvs, indices, entities, meshes, instances, transforms);
                            });
                        });
                    });
                });
            });
        });
    }), [&] {
        _positions.clear_cache();
        _normals.clear_cache();
        _tex_coords.clear_cache();
        _triangles.clear_cache();
        _entities.clear_cache();
        _instances.clear_cache();
        _instance_transforms.clear_cache();
    });
    
    _acceleration = device->build_acceleration(_positions, _triangles, meshes, _instances, _instance_transforms);
}

void Geometry::_encode(const std::vector<std::shared_ptr<Shape>> &shapes,
                       float3 *positions,
                       float3 *normals,
                       float2 *uvs,
                       packed_uint3 *triangles,
                       Entity *entities, std::vector<packed_uint3> &entity_ranges,
                       uint *instances,
                       float4x4 *instance_transforms) {
    
    auto vertex_offset = 0u;
    auto triangle_offset = 0u;
    auto entity_offset = 0u;
    
    // TODO
}

}
